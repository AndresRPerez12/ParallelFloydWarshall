// mpicxx -o MPI MPI.cpp `pkg-config --cflags --libs opencv`
// mpirun -np 4 MPI
#include <opencv2/highgui.hpp>
#include <bits/stdc++.h>
#include <sys/time.h>

#include "mpi.h"

using namespace std;

cv::Mat image;
cv::Mat original;
int boxSize;
int pRank, size;
int processRows, processColumns;

struct timeval tval_before, tval_after, tval_result;

int *processB;
int *processG;
int *processR;

string fileName;
string fileImage;
string outputFileName;

void averageBox ( int loY, int hiY, int loX , int hiX ){
    int blue_sum = 0;
    int green_sum = 0;
    int red_sum = 0;
    int counter = 0;
    for( int y = loY ; y <= hiY ; y ++ ){
        for( int x = loX ; x <= hiX ; x ++ ){
            blue_sum += processB[y*processColumns+x];
            green_sum += processG[y*processColumns+x];
            red_sum += processR[y*processColumns+x];;
            counter ++;
        }
    }
    int blue_avg = blue_sum/counter;
    int green_avg = green_sum/counter;
    int red_avg = red_sum/counter;
    for( int y = loY ; y <= hiY ; y ++ ){
        for( int x = loX ; x <= hiX ; x ++ ){
            processB[y*processColumns+x] = blue_avg;
            processG[y*processColumns+x] = green_avg;
            processR[y*processColumns+x] = red_avg;
        }
    }
}

void solveProcess( ){
    int initY = 0, endY = processRows-1;
    for( int y = initY ; y <= endY ; y += boxSize ){
        for( int x = 0 ; x < processColumns ; x += boxSize ){
            averageBox( y , min( y+boxSize-1 , endY ) , x , min( x+boxSize-1 , processColumns-1 ) );
        }
    }
}
 
int main(int argc, char* argv[]) {

    // string fileName = argv[1];
    // string fileImage = fileName + ".jpg";
    // string outputFileName = fileName + "_pixelated.jpg";
    
    // stringstream boxSizeSs(argv[2]);
    // boxSizeSs >> boxSize;

    // stringstream threadsSs(argv[3]);
    // threadsSs >> threads;

    MPI_Init( &argc, &argv );

        int root = 0;
        MPI_Comm_rank( MPI_COMM_WORLD, &pRank );
        MPI_Comm_size( MPI_COMM_WORLD, &size );

        int *blue;
        int *green;
        int *red;

        if( pRank == root ){
            fileName = "4k";
            fileImage = fileName + ".jpg";
            outputFileName = fileName + "_pixelated.jpg";
            boxSize = 20;

            original = cv::imread( fileImage ,cv::IMREAD_COLOR);
            image = cv::imread( fileImage ,cv::IMREAD_COLOR);
            
            if(! image.data ) {
                std::cout <<  "Image not found or unable to open" << std::endl ;
                return -1;
            }

            processRows = image.rows/size;
            processColumns = image.cols;

            blue = (int *)malloc( sizeof(int) * (image.rows*image.cols) );
            green = (int *)malloc( sizeof(int) * (image.rows*image.cols) );
            red = (int *)malloc( sizeof(int) * (image.rows*image.cols) );

            for( int y = 0 ; y < image.rows ; y ++ ){
                for( int x = 0 ; x < image.cols ; x ++ ){
                    cv::Vec3b intensity = image.at<cv::Vec3b>(y, x);
                    int pos = y * (image.cols) + x;
                    blue[pos] = int( intensity.val[0] );
                    green[pos] = int( intensity.val[1] );
                    red[pos] = int( intensity.val[2] );
                }
            }

            gettimeofday(&tval_before, NULL);
        }
        MPI_Barrier( MPI_COMM_WORLD );

        // enviar blockSize
        MPI_Bcast( &boxSize, 1, MPI_INT, root, MPI_COMM_WORLD); 
        // enviar cantidad de filas
        MPI_Bcast( &processRows, 1, MPI_INT, root, MPI_COMM_WORLD); 
        // enviar cantidad de columnas
        MPI_Bcast( &processColumns, 1, MPI_INT, root, MPI_COMM_WORLD); 
        MPI_Barrier( MPI_COMM_WORLD );
        // enviar filas de cada color
        int sendSize = processRows*processColumns;
        processB = (int *)malloc( sizeof(int) * sendSize );
        processG = (int *)malloc( sizeof(int) * sendSize );
        processR = (int *)malloc( sizeof(int) * sendSize );
        MPI_Scatter( blue, sendSize, MPI_INT, processB, sendSize, MPI_INT, root, MPI_COMM_WORLD );
        MPI_Scatter( green, sendSize, MPI_INT, processG, sendSize, MPI_INT, root, MPI_COMM_WORLD );
        MPI_Scatter( red, sendSize, MPI_INT, processR, sendSize, MPI_INT, root, MPI_COMM_WORLD );
        MPI_Barrier( MPI_COMM_WORLD );

        // ejecutar filtro
        solveProcess();
        MPI_Barrier( MPI_COMM_WORLD );

        // recibir filas de cada color
        MPI_Gather( processB, sendSize, MPI_INT, blue, sendSize, MPI_INT, root, MPI_COMM_WORLD );
        MPI_Gather( processG, sendSize, MPI_INT, green, sendSize, MPI_INT, root, MPI_COMM_WORLD );
        MPI_Gather( processR, sendSize, MPI_INT, red, sendSize, MPI_INT, root, MPI_COMM_WORLD );
        MPI_Barrier( MPI_COMM_WORLD );

        if( pRank == root ){
            gettimeofday(&tval_after, NULL);
            timersub(&tval_after, &tval_before, &tval_result);
            printf("%ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

            for( int y = 0 ; y < image.rows ; y ++ ){
                for( int x = 0 ; x < image.cols ; x ++ ){
                    int pos = y * (image.cols) + x;
                    cv::Vec3b intensity = image.at<cv::Vec3b>(y, x);
                    intensity.val[0] = uchar( blue[pos] );
                    intensity.val[1] = uchar( green[pos] );
                    intensity.val[2] = uchar( red[pos] );
                    image.at<cv::Vec3b>(y, x) = intensity;
                }
            }

            free( blue );
            free( green );
            free( red );

            // cv::namedWindow( "OpenCV original", cv::WINDOW_AUTOSIZE );
            // cv::imshow( "OpenCV original", original );
            // cv::namedWindow( "OpenCV Pixelada", cv::WINDOW_AUTOSIZE );
            // cv::imshow( "OpenCV Pixelada", image );
            cv::imwrite(outputFileName, image);
            cv::waitKey(0);
        }

        free( processB );
        free( processG );
        free( processR );

    MPI_Finalize( );
  return 0;
}