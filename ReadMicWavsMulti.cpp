#include <dirent.h>
#include <vector>
#include <algorithm>
#include <string>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/types.h>
#include <sys/dir.h>
#include <sys/param.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>

#include <sndfile.h>
#include <jack/jack.h>

// Include FFTW header
#include <complex>
#include <fftw3.h>

// #include <Eigen/Eigenvalues>
// Eigen include
#include <Eigen/Eigen>

#ifndef WIN32
#include <unistd.h>
#endif

using namespace std;

extern  int alphasort();

std::complex<double> *i_fft_1_ventana, *i_time_1_ventana,*i_fft_2_ventana, *i_time_2_ventana, *i_fft_3_ventana, *i_time_3_ventana,*i_fft_4_ventana, *i_time_4_ventana, *o_fft_4N, *o_time_4N;
std::complex<double> *i_fft_2N, *i_time_2N, *o_fft_2N, *o_time_2N;

bool READ_ENDED = false;

unsigned int sr_pcm;

double sample_rate  = 48000.0;			// default sample rate [Hz]
int nframes 		= 1024;	
int N = 3;
double d = 0.18; //DIstancia en metros
int c = 343; // Velocidad del sonido
int r = 2; // numero de señales en el subespacio

int window_size, window_size_2, nframes_2;
int kmin, kmax;	

double *freqs;

string app_name;
string channel_basename;
string wav_location;
vector<string> wavs_path;
vector<string> channel_names;

SNDFILE ** wavs;
SF_INFO *wavs_info;
int * wavs_i;
int lengthAngles;
std::vector<std::vector<double>> music_spectrum;
unsigned int channels;
unsigned int outputted_channels=0;
unsigned int outputted_channels_ids[100];

std::complex<double> *i_fft, *i_time, *o_fft, *o_time;
std::complex<double> frequencies[4][2048];
std::complex<double> finalFreqs[3][4][2048];
fftw_plan i_forward_1_ventana,i_forward_2_ventana,i_forward_3_ventana, i_forward_4_ventana, o_inverse;

jack_port_t *input_port;
// JACK:
jack_port_t **input_ports;

jack_port_t **output_port;
jack_client_t *client;

Eigen::VectorXd angles;
int fft_size;
int buffer_size;
jack_default_audio_sample_t **X_full;


void millisleep(int milli){
  struct timespec st = {0};
  st.tv_sec = 0;
  st.tv_nsec = milli*1000000L;
  nanosleep(&st, NULL);
}

Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> steeringVectors(int this_w){
  // Eigen::VectorXd angles(lengthAngles);
  // Compute steering vectors corresponding to values in angles
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> a1(N, lengthAngles);
    for (int k = 0; k < lengthAngles; k++) {
        a1(0, k) = 1.0; // First microphone is reference, no delay
        a1(1, k) = std::exp(std::complex<double>(0, -2 * M_PI * this_w * d / c * std::sin(angles[k] * M_PI / 180.0))); // Second mic, delayed one distance
        a1(2, k) = std::exp(std::complex<double>(0, -2 * M_PI * this_w * 2 * d / c * std::sin(angles[k] * M_PI / 180.0))); // Third mic, delayed double distance
    }

  return a1;
}

void music(const std::vector<std::vector<std::complex<double>>>& steeringVec, int f){
  for (int k = 0; k < lengthAngles; k++) {
    std::complex<double> numerator = 0.0;
    for (int i = 0; i < N; i++) {
        // numerator += std::conj(steeringVec[i][k]) * Qn * Qn * std::conj(steeringVec[i][k]);
    }
    music_spectrum[f][k] = std::abs(1.0 / numerator);
  }
}

static void signal_handler ( int sig ){
  jack_client_close ( client );
  printf ("ReadMicWavsMulti: signal received, exiting ...\n" );
  exit ( 0 );
}

int process ( jack_nframes_t jack_buffer_size, void *arg ) {
  //Initializing I/O variables
  unsigned int i,j;
  double read_buffer[channels][jack_buffer_size];
  int read_count[channels];
  bool ended = false;

  // índice de la frecuencia más baja que queremos procesar
  int idx_low = std::round(20 * fft_size / sample_rate);

  // índice de la frecuencia más alta que queremos procesar
  int idx_high = std::round(20000 * fft_size / sample_rate);
  
  //Writing to buffers
  jack_default_audio_sample_t *pDataOut[channels];

  jack_default_audio_sample_t **in;

	in = (jack_default_audio_sample_t **)malloc(channels*sizeof(jack_default_audio_sample_t *));
	for(i = 0; i < channels; ++i){
		in[i] = (jack_default_audio_sample_t *)jack_port_get_buffer (output_port[i], nframes);
  }

  // printf("fft_size: %d\n", fft_size);
  //Array de ventanas
  for (j = 0; j < channels; ++j) {

		for (i = 0; i < nframes; ++i){
			X_full[j][i] 						            = X_full[j][nframes + i];
			X_full[j][nframes + i] 	            = X_full[j][nframes*2 + i];
			X_full[j][nframes *2 + i] 	        = X_full[j][nframes * 3 + i];
			X_full[j][nframes * 3 + i] 	        = in[j][i];	
		}

	}
 
  //Convertir a Frecuencias
  for (int k = 0; k < channels; ++k){
    // ---------------------------- 1st window ------------------------------------------

    // FFT of the 1st window:
    for(i = 0; i < nframes; i++){
      i_time_1_ventana[i] = X_full[k][i];
    }
    fftw_execute(i_forward_1_ventana);

    // ---------------------------- 2nd window ------------------------------------------

    // FFT of the 2nd window:
    for(i = nframes; i < nframes*2; i++){
     i_time_2_ventana[i] = X_full[k][i];
    }
    fftw_execute(i_forward_2_ventana);

    // ---------------------------- 3rd window ------------------------------------------

    for(i = nframes*2; i < nframes*3; i++){
      i_time_3_ventana[i] = X_full[k][i];
    }
    fftw_execute(i_forward_3_ventana);

    // ---------------------------- 4th window ------------------------------------------

    for(i = nframes*3; i < window_size; i++){
      i_time_4_ventana[i] = X_full[k][i];
    }
    fftw_execute(i_forward_4_ventana);


    // Asignar valores a la matriz
    for (int j = 0; j <4; j++) {
      for(int i = 0; i <fft_size; ++i){
        if(j==0){
          frequencies[j][i] = i_fft_1_ventana[i];
        }
        if(j==1){
          frequencies[j][i] = i_fft_2_ventana[i];
        }
        if(j==2){
          frequencies[j][i] = i_fft_3_ventana[i];
        }
        if(j==3){
          frequencies[j][i] = i_fft_4_ventana[i];
        }
      }  
         
    }

    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < fft_size; ++i) {
            finalFreqs[k][j][i] = frequencies[j][i];
        }
    }

	}

  for (int i = 0; i < fft_size; ++i) {
    std::vector<std::vector<std::complex<double>>> matriz2(3, std::vector<std::complex<double>>(4));

    Eigen::MatrixXcd matriz(3,4);
    // std::complex<double> value = finalFreqs[k][j][i];

    matriz(0,0) = finalFreqs[0][0][i];
    matriz(0,1) = finalFreqs[0][1][i];
    matriz(0,2) = finalFreqs[0][2][i];
    matriz(0,3) = finalFreqs[0][3][i];


    matriz(1,0) = finalFreqs[1][0][i];
    matriz(1,1) = finalFreqs[1][1][i];
    matriz(1,2) = finalFreqs[1][2][i];
    matriz(1,3) = finalFreqs[1][3][i];


    matriz(2,0) = finalFreqs[2][0][i];
    matriz(2,1) = finalFreqs[2][1][i];
    matriz(2,2) = finalFreqs[2][2][i];
    matriz(2,3) = finalFreqs[2][3][i];

    // Calcular matriz de covarianza
    std::cout << "--- Complex Number Matrix C:\n" << matriz << std::endl << std::endl;

    std::cout << "Freqs\n" << freqs[200] << std::endl;
    std::cout << "Freqs22\n" << 400/sample_rate*nframes << std::endl;
    // Eigen::MatrixXcd matriz_2(3,4);
    // matriz_2 = matriz.adjoint();

    // std::cout << "Matriz de matriz_2:\n" << matriz_2 << std::endl;

    std::cout << "--- Matriz.adjoint()' :\n" << matriz.adjoint() << std::endl << std::endl;

    Eigen::MatrixXcd covarianza = matriz * matriz.adjoint();
    std::cout << "Matriz de covarianza:\n" << covarianza << std::endl;


    //Sacamos los eigenvalores eigenvectores
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigenM(covarianza);
    std::cout << "--- The eigenvalues of eigenM are:\n" << eigenM.eigenvalues() << std::endl << std::endl;
    std::cout << "--- The eigenvectors of eigenM are (one vector per column):\n" << eigenM.eigenvectors() << std::endl << std::endl;
	  // std::cout << "--- The first eigenvector:\n" << eigenM.eigenvectors().col(0) << std::endl << std::endl;

    // Obtener los eigenvalores y eigenvectores calculados
    Eigen::VectorXcd eigenvalues = eigenM.eigenvalues();
    Eigen::MatrixXcd eigenvectors = eigenM.eigenvectors();


    // std::cout << "Eigenvalores descendentes:\n";
    // for (const auto& eigenvalue : eigenvalues) {
    //     std::cout << eigenvalue << std::endl;
    // }

    // Actualizar los eigenvectores correspondientes
    // for (int i = 0; i < eigenvectors.cols(); i++) {
    //     int idx = std::distance(eigenvalues.real().data(), std::find(eigenvalues.real().data(), eigenvalues.real().data() + eigenvalues.size(), eigenvalues_vec[i]));
    //     eigenvectors.col(i) = eigenvectors.col(idx);
    // }

    // std::cout << "Eigenvectores correspondientes:\n";
    // for (const auto& eigenvalue : eigenvalues) {
    //     int idx = std::distance(eigenvalues.real().data(), std::find(eigenvalues.real().data(), eigenvalues.real().data() + eigenvalues.size(), eigenvalue));
    //     std::cout << "Eigenvalor: " << eigenvalue << std::endl;
    //     std::cout << "Eigenvector:\n" << eigenvectors.col(idx) << std::endl;
    // }

    // // Obtener los eigenvectores de la señal (primeros r eigenvectores ordenados)
    // // Obtener los eigenvectores del ruido (eigenvectores restantes)

    

    Eigen::MatrixXcd Qs(eigenvectors.rows(), r );
    // std::vector<std::vector<std::complex<double>>> Qs(eigenvectors.size(), std::vector<std::complex<double>>(r));

    Eigen::MatrixXcd Qn(eigenvectors.rows(), eigenvectors.cols() - r );
    // std::vector<std::vector<std::complex<double>>> Qn(eigenvectors.size(), std::vector<std::complex<double>>(eigenvectors.cols() - r));
    Eigen::VectorXcd firstEigenvector = eigenvectors.col(0);

    for (int i = 0; i < eigenvectors.rows(); ++i) {
        for (int j = 0; j < eigenvectors.cols(); ++j) {
          if(j < r){
            Qs(i,j) = eigenvectors(i,j);
          }else{
            Qn(i,j-r) = eigenvectors(i,j);
          }
        }
    }

    std::cout << "Matriz Qs:" << std::endl;
    std::cout << Qs << std::endl;

    std::cout << "Matriz Qn:" << std::endl;
    std::cout << Qn << std::endl;

    

    // double degrees = 150 - angles[k];
    // double radians = degrees * M_PI / 180.0; // Convertir grados a radianes
    // double t3 = (d / c) * cos(radians);

    // Llamada a la función steeringVectors
    Eigen::MatrixXcd steeringVec(N,lengthAngles);
    // std::vector<std::vector<std::complex<double>>> steeringVec(N, std::vector<std::complex<double>>(lengthAngles));
    for (int k = 0; k < lengthAngles; k++) {
        double t3 = (int) (d/c)* cos(150 - angles[k]);
        steeringVec(0 ,k) = std::complex<double>(1.0, 0.0); // First microphone is reference, no delay
        steeringVec(1,k) = std::exp(std::complex<double>(0, -2 * M_PI * freqs[i] * d / c * sin(angles[k] * M_PI / 180.0))); // Second mic, delayed one distance
        steeringVec(2,k) = std::exp(std::complex<double>(0, -2 * M_PI * freqs[i] * 2 * t3 * sin(angles[k] * M_PI / 180.0))); // Third mic, delayed double distance
    }


    std::cout << "steeringVector completo:" << std::endl;
    std::cout << steeringVec.col(0) << std::endl;

    std::cout << "steeringVector completo:" << std::endl;
    std::cout << steeringVec.col(0).adjoint() << std::endl;
        
    Eigen::MatrixXcd producto = Qn * Qn.adjoint();

    std::cout << "Matriz producto:" << std::endl;
    std::cout << producto << std::endl;

    // // Qn * Qn´
    // // 400 3000  
    // //hacerlo con eigen 
    // //plotear las frecuencias y ver cuales salen?
    std::complex<double> num(1.0, 0.0);
    // MUSIC
    // music_spectrum(f,k)=abs(1/(a1(:,k)'*Qn*Qn'*a1(:,k)));
    // Problema al multiplicar steeringVec * producto
    for (int k = 0; k < lengthAngles; k++) {
      // Music en la frecuencia i y angulo k
      // Eigen::RowVectorXcd
      // double current_music_value = std::real((a1.adjoint() * a1)(0,0)) / std::real((a1.adjoint()*Qn*Qn.adjoint()*a1)(0,0));
      // music_spectrum[i][k] = std::abs(1.0 / (steeringVec.col(k).adjoint() * producto * steeringVec.col(k)));
    }




    // Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> result = steeringVectors(freqs[i]);



  }



  for (j=0;j<channels;j++){
    pDataOut[j] = (jack_default_audio_sample_t*)jack_port_get_buffer ( output_port[j], jack_buffer_size );
    read_count[j] = sf_read_double(wavs[j],read_buffer[j],jack_buffer_size);
  }
  // printf("jack_buffer_size %d",jack_buffer_size);
  for (i=0;i<jack_buffer_size;i++){
    for (j=0;j<channels;j++){
      if (read_count[j] != jack_buffer_size && i >= read_count[j]){
        // Finished reading file
        // Completing the buffer with silence
        pDataOut[j][i] = 0.0;
        
        if (ended == false){
          cout << "ReadMicWavsMulti: Finished playing." << endl;
          ended = true;
        }
      }else{
        pDataOut[j][i] = (jack_default_audio_sample_t)read_buffer[j][i];
      }
      wavs_i[j]++;
    }
  }
  
  if (ended == true){
    cout << "ReadMicWavsMulti: Finished playing." << endl;
    READ_ENDED = true;
  }
  
  return 0;      
}

void jack_shutdown ( void *arg ){
  free ( output_port );
  exit ( 1 );
}

int file_select(const struct dirent *entry){
  char *ptr;
  //char *rindex(const char *s, char c);
  
  if ((strcmp(entry->d_name, ".")== 0) || (strcmp(entry->d_name, "..") == 0))
    return 0;
  
  /* Check for filename extensions */
  ptr = rindex((char *)entry->d_name, '.');
  if ((ptr != NULL) && (strcmp(ptr, ".wav") == 0))
    return 1;
  else
    return 0;
}

void usage(){
  cout << "Usage: ReadMicWavsMulti app_name channel_basename wav_location number_channels input_channel0 input_channel1 ... " << endl;
  cout << "Usage: ReadMicWavsMulti ReadMicWavs wav_mic ./wav_mics 3 0 1 2 " << endl;
  cout << "\t app_name: name of JACK client where the outputs will connect to" << endl;
  cout << "\t channel_basename: basename of client inputs" << endl;
  cout << "\t wav_location: directory where wav_micX.wav's are located" << endl;
  cout << "\t number_channels: number of channels to connect to JACK client" << endl;
  cout << "\t input_channelX: channel number to connect X output" << endl;
}

int main ( int argc, char *argv[] ){
  if (argc < 5){
    usage();
    exit(1);
  }
  
  app_name = string(argv[1]);
  channel_basename = string(argv[2]);
  wav_location = string(argv[3]);
  channels = atoi(argv[4]);
  
  if (channels > 16){
    usage();
    exit(1);
  }
  
  cout << "ReadMicWavsMulti: Probing app   : " << app_name << endl;
  cout << "ReadMicWavsMulti: With " << channels << " channels with basename: " << channel_basename << "X" << endl;
  cout << "ReadMicWavsMulti: Reading from  : " << wav_location << endl;
  
  int i,j,k;
  
  /***********READING STUFF************/
  /* Obtain the WAV file list */
  wavs_i = (int *)malloc(channels * sizeof(int));
  wavs = (SNDFILE **)malloc(channels * sizeof(SNDFILE *));
  wavs_info = (SF_INFO *)malloc(channels * sizeof(SF_INFO));
  
  for (j=0; j<channels; j++){
    wavs_i[j] = 0;
    wavs_path.push_back(string(wav_location)+string("/wav_mic")+to_string(j+1)+string(".wav"));
    wavs_info[j].format = 0;
    
    cout << "ReadMicWavsMulti: Output " << j+1 <<" -> opening " << wavs_path[j] << endl;
    wavs[j] = sf_open (wavs_path[j].c_str(),SFM_READ,&wavs_info[j]);
    if (wavs[j] == NULL){
      printf ("ReadMicWavsMulti: Output %d -> Could not open '%s'\n", j+1, wavs_path[j].c_str()) ;
      exit(1);
    }
  }
  
  for (i=0,j=5; j<argc; j++,i++){
    int channel_id = atoi(argv[j]);
    outputted_channels_ids[i] = channel_id;
    //if channel_id is invalid ("0" or negative), it will be ignored by "jack_connect" later on
    channel_names.push_back(string(app_name)+string(":")+string(channel_basename)+to_string(channel_id));
    printf("channel_id %d",channel_id);
    printf("channels %d",channels);
    if(channel_id > channels || channel_id <= 0){
      printf("ReadMicWavsMulti: Output %d will not be connected; channel input %d is outside range [1,%d].\n",i+1, channel_id,channels);
    }else{
      printf("ReadMicWavsMulti: Output %d will be connected to channel input %d.\n",i+1,channel_id);
    }
  }
  
  /***********JACK STUFF************/
  const char *client_name = "ReadMicWavs";
  const char *server_name = NULL;
  jack_options_t options = JackNullOption;
  jack_status_t status;
  
  client = jack_client_open ( client_name, options, &status, server_name );
  if ( client == NULL ){
    printf ("ReadMicWavsMulti: jack_client_open() failed, "
              "status = 0x%2.0x\n", status );
    if ( status & JackServerFailed ){
        printf ("ReadMicWavsMulti: Unable to connect to JACK server\n" );
    }
    exit ( 1 );
  }
  if ( status & JackNameNotUnique ){
      client_name = jack_get_client_name ( client );
      printf ("ReadMicWavsMulti: unique name `%s' assigned\n", client_name );
  }
  jack_set_process_callback ( client, process, 0 );
  jack_on_shutdown ( client, jack_shutdown, 0 );
  sr_pcm = (unsigned int) jack_get_sample_rate (client);
  cout << "ReadMicWavsMulti: JACK sample rate : "<< sr_pcm << "." << endl;
  cout << "ReadMicWavsMulti: JACK buffer size : "<< jack_get_buffer_size(client) << "." << endl;

  sample_rate = (double)jack_get_sample_rate(client);
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
  int cnt_tst = 0;

  for (i = 0; i <= nframes/2; ++i) {
		if (freqs[i] >= 40 && freqs[i] <= 8000){
			printf("I: %d // CNT: %d // FREQ = %lf\n", i, cnt_tst, freqs[i]);
			cnt_tst++;
		}
	}

 

    const double angle_min = -90.0;
    const double angle_max = 90.0;
    const int num_angles = (angle_max - angle_min) / 0.1 + 1;

    angles = Eigen::VectorXd::LinSpaced(num_angles, angle_min, angle_max);


  lengthAngles = angles.size();
  music_spectrum = std::vector<std::vector<double>>(r, std::vector<double>(lengthAngles, 0.0));
  // obtain here the delay from user and store it in 'delay' 
	nframes 	= (int) jack_get_buffer_size (client);
	nframes_2   = nframes/2;
	window_size = 4*nframes;
	window_size_2 = 2*nframes;
	// kmin = (int) (f_min/sample_rate*window_size_2);
	// kmax = (int) (f_max/sample_rate*window_size_2);

  // initialization of internal buffers
	// - overlap-add buffers
	// X_late		= (jack_default_audio_sample_t **) calloc(n_in_channels, sizeof(jack_default_audio_sample_t*));
	// X_early		= (jack_default_audio_sample_t **) calloc(n_in_channels, sizeof(jack_default_audio_sample_t*));
	X_full		= (jack_default_audio_sample_t **) calloc(channels, sizeof(jack_default_audio_sample_t*));


   //preparing FFTW3 buffers
  i_time_1_ventana = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * fft_size);
  i_time_2_ventana = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * fft_size);
  i_time_3_ventana = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * fft_size);
  i_time_4_ventana = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * fft_size);

  // i_time = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * fft_size);
  i_fft_1_ventana = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * fft_size);

  i_fft_2_ventana = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * fft_size);

  i_fft_3_ventana = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * fft_size);

  i_fft_4_ventana = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * fft_size);
  //o_time = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * fft_size);

  i_forward_1_ventana = fftw_plan_dft_1d(window_size, reinterpret_cast<fftw_complex*>(i_time_1_ventana), reinterpret_cast<fftw_complex*>(i_fft_1_ventana), FFTW_FORWARD, FFTW_MEASURE);

  i_forward_2_ventana = fftw_plan_dft_1d(window_size, reinterpret_cast<fftw_complex*>(i_time_2_ventana), reinterpret_cast<fftw_complex*>(i_fft_2_ventana), FFTW_FORWARD, FFTW_MEASURE);

  i_forward_3_ventana = fftw_plan_dft_1d(window_size, reinterpret_cast<fftw_complex*>(i_time_3_ventana), reinterpret_cast<fftw_complex*>(i_fft_3_ventana), FFTW_FORWARD, FFTW_MEASURE);

  i_forward_4_ventana = fftw_plan_dft_1d(window_size, reinterpret_cast<fftw_complex*>(i_time_4_ventana), reinterpret_cast<fftw_complex*>(i_fft_4_ventana), FFTW_FORWARD, FFTW_MEASURE);






  // o_inverse = fftw_plan_dft_1d(window_size, reinterpret_cast<fftw_complex*>(o_fft), reinterpret_cast<fftw_complex*>(o_time), FFTW_BACKWARD, FFTW_MEASURE);


  for (i = 0; i < channels; ++i) {
		X_full[i]	= (jack_default_audio_sample_t *) calloc(window_size, sizeof(jack_default_audio_sample_t));

    // X_full[i]	= (jack_default_audio_sample_t *) calloc(window_size + window_size_2, sizeof(jack_default_audio_sample_t));
    
  }	
  
  char port_name[16];
  output_port = ( jack_port_t** ) malloc ( channels*sizeof ( jack_port_t* ) );
  for ( i = 0; i < channels; i++ ){
    
    sprintf ( port_name, "out_%d", i+1 );
    printf ("ReadMicWavsMulti: registering port %s \n", port_name);
    output_port[i] = jack_port_register ( client, port_name, JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0 );
    if ( output_port[i] == NULL ){
      printf ("ReadMicWavsMulti: no more JACK ports available\n" );
      exit ( 1 );
    }
  }

  // input_ports = (jack_port_t**) malloc(channels*sizeof(jack_port_t*));
	// for(i = 0; i < channels; ++i) {
	// 	// sprintf(portname, "wav_mic%d", i+1);
	// 	input_ports[i] = jack_port_register (client, port_name, JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
	// 	if (input_ports[i] == NULL) {
	// 		printf("No more JACK ports available after creating input port number %d\n",i);
	// 		exit (1);
	// 	}
	// }	
  
  
  if ( jack_activate ( client ) ){
    printf ("ReadMicWavsMulti: cannot activate client" );
    exit ( 1 );
  }
  
  /* Connect app to these outputs */
  cout << "ReadMicWavsMulti: Connecting our outputs to " << app_name.c_str() << " inputs." << endl;
  for (i = 0; i < channels; i++){
    if (outputted_channels_ids[i] <= channels && outputted_channels_ids[i] > 0){
      cout << "ReadMicWavsMulti: Connecting " << jack_port_name (output_port[i]) << " to " << channel_names[i].c_str() << endl;
      if (jack_connect (client, jack_port_name (output_port[i]), channel_names[i].c_str()))
        printf ("ReadMicWavsMulti: cannot connect input ports.\n" );
    }
  }
  
  /* install a signal handler to properly quits jack client */
#ifdef WIN32
  signal ( SIGINT, signal_handler );
  signal ( SIGABRT, signal_handler );
  signal ( SIGTERM, signal_handler );
#else
  signal ( SIGQUIT, signal_handler );
  signal ( SIGTERM, signal_handler );
  signal ( SIGHUP, signal_handler );
  signal ( SIGINT, signal_handler );
#endif
  
  /* keep running until the transport stops */
  
  while (!READ_ENDED){
#ifdef WIN32
    Sleep ( 1000 );
#else
    sleep ( 1 );
#endif
  }
  
  free ( output_port );
  
  jack_client_close ( client );
  exit ( 0 );
}
