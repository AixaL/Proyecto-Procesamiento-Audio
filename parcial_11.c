#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <jack/jack.h>

#include <complex.h>
#include <fftw3.h>

jack_port_t *input_port_1; 
jack_port_t *output_port_1;
jack_port_t *output_port_2;
jack_client_t *client; 
jack_default_audio_sample_t *buffer;
jack_default_audio_sample_t buffer_2[1024];

fftw_plan i_forward, o_inverse , buffer2_inverse, buffer2_forward ;
// fftw_plan i_forward, o_inverse, buffer1_inverse, buffer2_forward, buffer1_forward ,buffer2_inverse;

double complex *i_fft, *i_time, *o_fft, *o_time, *buffer1, *buffer2, *buffer2_i_time, *buffer2_i_fft, *buffer2_o_fft, *buffer2_o_time, *buffer1_time, *buffer2_time,  *buffer1_fft, *buffer2_fft; 
int buffer_size;
int buffer_i_1 = 0;
int buffer_i_2 = 0;
float segundos = 0.001;
double sample_rate;
double *freqs;
double cos(double x);
double *hann;
int fft_size;
double complex cexp(double complex z);

double PI = 3.14159265358979323846;


//void filter(jack_default_audio_sample_t *b){

//}

int jack_callback (jack_nframes_t nframes, void *arg){
  jack_default_audio_sample_t *in, *out_1, *in_2 , *out_2; 
  int i;
  
  in = jack_port_get_buffer (input_port_1, nframes);
  out_1 = jack_port_get_buffer (output_port_1, nframes); 
  out_2 = jack_port_get_buffer (output_port_2, nframes);


  for (i = 0; i < nframes; ++i)
  {
    //out_1[i] = in[i];
    out_1[i] = buffer_2[i];
    buffer_2[i] = in[i]; 
  }

  //Asumimos que :
  //b1 tiene las dos ultimas ventanas hanneadas
  //b2 tiene la ultima ventana en su primera mitad (ventana pasada)

  // señal in ---> Buffer2 segunda parte
  for(i =  0; i < nframes ; ++i){

    buffer2_i_time[nframes + i] = in[i];
    
  }

//Filtrar b2 / Hacerlo en una función
// filter(b2);

//HANN
 for(i = 0; i < fft_size; ++i){
   double multiplier = sqrt( 0.5 * (1.0 - cos(2.0*PI*i/(fft_size))));
   buffer2_i_time[i] = multiplier * buffer2_i_time[i];
  }


  fftw_execute(i_forward);
  //fftw_execute(buffer2_forward);

//Vamos a dominio de la frecuencia
//Aplicamos el desfase
 for(i = 0; i < fft_size; ++i){
   buffer2_o_fft[i] = buffer2_i_fft[i]*cexp(-I*2.0*PI*freqs[i]*segundos);

  }

  fftw_execute(o_inverse);
  //fftw_execute(buffer2_inverse);
  //Regresamos al dominio del tiempo

  //HANN

  for(i = 0; i < fft_size; ++i){
      double multiplier = sqrt( 0.5 * (1.0 - cos(2.0*PI*i/(fft_size))));

      buffer2_o_time[i] = multiplier * (buffer2_o_time[i])/fft_size;
  }

  for (i = 0; i < nframes; i++) {
      out_2[i] = buffer2_o_time[i] + buffer1_time[nframes + i];
  }

//Pasar Buffer2 -----> Buffer1
  for (i = 0; i < nframes*2; ++i)
  {
    buffer1_time[i] = buffer2_o_time[i];
    //buffer2_i_time[i] = buffer2_o_time[i];
  } 

  //Pasar =  in ---> Buffer2 primera Parte

  for(i = 0; i < nframes; ++i){
    
    buffer2_i_time[i] = in[i];
    
  }


  return 0;
}


/**
 * JACK calls this shutdown_callback if the server ever shuts down or
 * decides to disconnect the client.
 */
void jack_shutdown (void *arg){
  exit (1);
}


int main (int argc, char *argv[]) {

 
    //if(argc < 2){
      //exit (1);
    //}

  //printf("%s",argc);  

  printf("Entro");

  //segundos= 0.05;
  //segundos= atof(argv[1]);
  //tiempo de desfase
  

  const char *client_name = "in_to_out";
  jack_options_t options = JackNoStartServer; 
  jack_status_t status;
  
  
  client = jack_client_open (client_name, options, &status); 
  

  if (client == NULL){
   
    printf ("jack_client_open() failed, status = 0x%2.0x\n", status);
    if (status & JackServerFailed) {
      printf ("Unable to connect to JACK server.\n");
    }
    exit (1);
  }
  
  if (status & JackNameNotUnique){
    //Cambio el nombre del agente. Más de uno con el mismo nombre
    client_name = jack_get_client_name(client);
    printf ("Warning: other agent with our name is running, `%s' has been assigned to us.\n", client_name);
  }
  
  /* tell the JACK server to call 'jack_callback()' whenever there is work to be done. */
  jack_set_process_callback (client, jack_callback, 0);
  
  jack_on_shutdown (client, jack_shutdown, 0);
 
   /* display the current sample rate. */
  printf ("Sample rate: %d\n", jack_get_sample_rate (client));
  printf ("Window size: %d\n", jack_get_buffer_size (client));
  sample_rate = (double)jack_get_sample_rate(client); // Frecuencia de muestreo ? Hz
  int nframes = jack_get_buffer_size (client);
  

  // prepare frecuency array
  freqs = (double *) malloc(sizeof(double) * nframes);
  freqs[0] = 0.0;
  double f1 = sample_rate/nframes; 
  //48000/1024

  fft_size = nframes * 2;
  buffer_size = nframes * 2;

  for (int i = 1; i < nframes/2; ++i)
  {
    freqs[i] = f1 * i;
    freqs[nframes-i] = -freqs[i];
  }

  freqs[nframes/2] = sample_rate/2;


  //hann malloc

  //VAD investigar uno Voice Activity Detection

  //Filtro pre-enfasis / en el dominio del tiempo para ir al dominio de la frecuencia -> clasificación

  //para poner los positivos y los negativos

  //preparing FFTW3 buffers
  buffer2_i_fft = (double complex *) fftw_malloc(sizeof(double complex ) * fft_size);
  buffer2_o_fft = (double complex *) fftw_malloc(sizeof(double complex ) * fft_size);

  buffer2_i_time = (double complex *) fftw_malloc(sizeof(double complex ) * fft_size);
  buffer2_o_time = (double complex *) fftw_malloc(sizeof(double complex ) * fft_size); //tamaño nframes. LO podemos cambiar que sea mayor?

  buffer1 = (double complex *) fftw_malloc(sizeof(double complex ) * (fft_size));
  buffer2 = (double complex *) fftw_malloc(sizeof(double complex ) * (fft_size));

  buffer2_time = (double complex *) fftw_malloc(sizeof(double complex ) * (fft_size));
  buffer1_time = (double complex *) fftw_malloc(sizeof(double complex ) * (fft_size));
  buffer2_fft = (double complex *) fftw_malloc(sizeof(double complex ) * (fft_size));

    // for (int i = 0; i < fft_size; ++i)
    // {
    //   buffer1[i]=0.0;
    //   buffer2[i]=0.0;
    // }
  
  i_forward = fftw_plan_dft_1d(fft_size, buffer2_i_time, buffer2_i_fft , FFTW_FORWARD, FFTW_MEASURE);
  o_inverse = fftw_plan_dft_1d(fft_size, buffer2_o_fft , buffer2_o_time, FFTW_BACKWARD, FFTW_MEASURE); //frecuencia a tiempo. asignar bien el tamaño (nframes)


  //buffer1_forward = fftw_plan_dft_1d(fft_size, buffer1_time, buffer1_fft , FFTW_FORWARD, FFTW_MEASURE);
  buffer2_forward = fftw_plan_dft_1d(fft_size, buffer2_time, buffer2_fft , FFTW_FORWARD, FFTW_MEASURE);


  //buffer1_inverse = fftw_plan_dft_1d(fft_size, buffer1_fft , buffer1_time, FFTW_BACKWARD, FFTW_MEASURE); 
  buffer2_inverse = fftw_plan_dft_1d(fft_size, buffer2_fft , buffer2_time, FFTW_BACKWARD, FFTW_MEASURE); 
  
  /* create the agent input port */
  input_port_1 = jack_port_register (client, "input", JACK_DEFAULT_AUDIO_TYPE,JackPortIsInput, 0);
  
  /* create the agent output port */
  output_port_1 = jack_port_register (client, "output",JACK_DEFAULT_AUDIO_TYPE,JackPortIsOutput, 0);
  output_port_2 = jack_port_register (client, "output_2",JACK_DEFAULT_AUDIO_TYPE,JackPortIsOutput, 0);

  /* check that both ports were created succesfully */
  if ((input_port_1 == NULL) || (output_port_1 == NULL) || (output_port_2 == NULL)){
    printf("Could not create agent ports. Have we reached the maximum amount of JACK agent ports?\n");
    exit (1);
  }

  if (jack_activate (client)) {
    printf ("Cannot activate client.");
    exit (1);
  }
  //inicia a conectarse con jack_callback

  
  printf ("Agent activated.\n");
  
  printf ("Connecting ports... ");
   
  /* Assign our input port to a server output port*/
  // Find possible output server port names
  const char **serverports_names;
  serverports_names = jack_get_ports (client, NULL, NULL, JackPortIsPhysical|JackPortIsOutput);
  //serie de strings se guardan los nombres de los outputs
  //Microfono
  if (serverports_names == NULL) {
    printf("No available physical capture (server output) ports.\n");
    exit (1);
  }
  // Connect the first available to our input port
  if (jack_connect (client, serverports_names[0], jack_port_name (input_port_1))) {
    printf("Cannot connect input port.\n");
    exit (1);
  }


  // free serverports_names variable for reuse in next part of the code
  free (serverports_names);
  
  
  /* Assign our output port to a server input port*/
  // Find possible input server port names
  serverports_names = jack_get_ports (client, NULL, NULL, JackPortIsPhysical|JackPortIsInput);
  //Bocinas
  if (serverports_names == NULL) {
    printf("No available physical playback (server input) ports.\n");
    exit (1);
  }
  // Connect the first available to our output port
  if (jack_connect (client, jack_port_name (output_port_1), serverports_names[0])) {
    printf ("Cannot connect output ports.\n");
    exit (1);
  }

  if (jack_connect (client, jack_port_name (output_port_2), serverports_names[1])) {
    printf ("Cannot connect output ports.\n");
    exit (1);
  }
  // free serverports_names variable, we're not going to use it again
  free (serverports_names);
  
  
  printf ("done.\n");
  /* keep running until stopped by the user */
  sleep (-1);
  
  
  /* this is never reached but if the program
     had some other way to exit besides being killed,
     they would be important to call.
  */
  jack_client_close (client);
  exit (0);
}

// make 
// gcc -o jack_in_to_out desfase2.c -ljack
//  gcc -o 2Baddies.c -ljack
// ./jack_in_to_out