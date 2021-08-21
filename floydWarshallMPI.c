// mpicc -o floydWarshallMPI floydWarshallMPI.c
// mpirun -np 4 floydWarshallMPI
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <limits.h>

#include <sys/time.h>

#include "mpi.h"

#define MAX_NODES 5000

struct timeval tval_before, tval_after, tval_result;
int distance [MAX_NODES*MAX_NODES];
int distanceBuffer [MAX_NODES*MAX_NODES];
int isInfiniteBuffer [MAX_NODES*MAX_NODES];
int isInfinite [MAX_NODES*MAX_NODES];

int nodes, edges;
int rank, size;
int printFlag;

void updateDistance( int lo_i , int hi_i , int k ){
    for( int i = lo_i ; i <= hi_i && i < nodes ; i ++ ){
        for( int j = 0 ; j < nodes ; j ++ ){
            int p_ij = i * nodes + j;
            int p_ik = i * nodes + k;
            int p_kj = k * nodes + j;
            if( isInfinite[p_ik] == 1 || isInfinite[p_kj] == 1 ) continue;
            if( isInfinite[p_ij] == 1 || distance[p_ij] > distance[p_ik] + distance[p_kj] ){
                distance[p_ij] = distance[p_ik] + distance[p_kj];
                isInfinite[p_ij] = 0;
                // printf("rank = %d k = %d :: dist[%d][%d] = %d\n",rank,k,i,j,distance[p_ij]);
            }
        }
    }
}

void solveThread( int threadId , int k ){
    int lo_i, hi_i;
    lo_i = ((nodes+size-1)/size) * threadId;
    hi_i = lo_i + (((nodes+size-1)/size)-1);
    if( threadId == size-1 ) hi_i = nodes;
    if( lo_i >= nodes ) return;
    updateDistance( lo_i , hi_i , k );
}

int main(int argc, char **argv){

    MPI_Init( &argc, &argv );
        int root = 0;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        MPI_Comm_size( MPI_COMM_WORLD, &size );

        if( rank == root ){
            freopen("input.txt","r",stdin);
            scanf("%d%d",&nodes,&edges);
            for( int i = 0 ; i < nodes ; i ++ ){
                for( int j = 0 ; j < nodes ; j ++ ){
                    if( i == j ){
                        isInfinite[i * nodes + j] = 0;
                        distance[i * nodes + j] = 0;
                    }else{
                        isInfinite[i * nodes + j] = 1;
                        distance[i * nodes + j] = INT_MAX;
                    }
                }
            }
            for( int i = 0 ; i < edges ; i ++ ){
                int nodeA, nodeB, weight;
                scanf("%d%d%d",&nodeA,&nodeB,&weight);
                nodeA --;
                nodeB --;
                isInfinite[nodeA*nodes + nodeB] = 0;
                distance[nodeA*nodes + nodeB] = weight;
            }
        }

        if( rank == root ){
            gettimeofday(&tval_before, NULL);
        }
        MPI_Bcast( &nodes, 1, MPI_INT, root, MPI_COMM_WORLD); 
        for( int k = 0 ; k < nodes ; k ++ ){
            // printf("rank: %d size: %d\n",rank,size);
            // mandar la matriz actual
            MPI_Bcast( distance, nodes*nodes, MPI_INT, root, MPI_COMM_WORLD);
            MPI_Bcast( isInfinite, nodes*nodes, MPI_INT, root, MPI_COMM_WORLD); 
            // ejecutar algo paralelo
            solveThread( rank , k );
            // recibir datos
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Reduce(distance, distanceBuffer, nodes*nodes, MPI_INT, MPI_MIN, root, MPI_COMM_WORLD);
            MPI_Reduce(isInfinite, isInfiniteBuffer, nodes*nodes, MPI_INT, MPI_MIN, root, MPI_COMM_WORLD);
            if( rank == root ){
                for( int i = 0 ; i < nodes ; i ++ ){
                    for( int j = 0 ; j < nodes ; j ++ ){
                        distance[i*nodes+j] = distanceBuffer[i*nodes+j];
                        isInfinite[i*nodes+j] = isInfiniteBuffer[i*nodes+j];
                    }
                }
            }
            // MPI Barrier
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if( rank == root ){
            gettimeofday(&tval_after, NULL);
            timersub(&tval_after, &tval_before, &tval_result);
            printf("%ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
            // freopen("outOpenMP.txt","w",stdout);
            // for( int i = 0 ; i < nodes ; i ++ ){
            //     for( int j = 0 ; j < nodes ; j ++ ){
            //         if( isInfinite[i*nodes+j] ) printf("-1 ");
            //         else printf("%d ",distance[i*nodes+j]);
            //     }
            //     printf("\n");
            // }
        }
    MPI_Finalize( );

}