/**
 * Author: Nik Zulhilmi Nik Fuaad
 * 		Computer Science/Software Engineering
 * 		University of Birmingham
 *
 * Forward Kinematics and Inverse Kinematics in Parallel
 *
 * For forward kinematics :-
 * Input format:
 * 1. Number of bodies, n - int
 * 2. Joint types (int array).
 * 		-0 for revolute
 * 		-1 for prismatic
 * 3. Q (float array)
 * 		-n-dimensional vector of joint variables.
 * 4. Denavit-Hartenberg Parameters (2d float array)
 * 		-nx4 matrix.
 * 5. PDEs, desired position - (2d int array)
 * 		-2x1 for planar robot
 * 		-3x1 for spatial robot
 * 6. ODEs, desired orientation - (2d int array)
 * 		-scalar for planar
 * 		-3x1 for 3D robot
 * 7. Method (int)
 * 		-value: 0 or 1
 *
 * This code calculates the forward kinematics of a manipulator.
 * It calculates the homogeneous transformation matrix from the base to the
 * end effector (T), and also the end-effector Jacobian with respect to
 * the base frame of a manipulator robot.
 *
 *
 * For inverse kinematics :-
 * Input format:
 * 1. Number of bodies, n - int
 * 2. Joint types (int array).
 * 		-0 for revolute
 * 		-1 for prismatic
 * 3. Q (float array)
 * 		-n-dimensional vector of joint variables.
 * 4. Denavit-Hartenberg Parameters (2d float array)
 * 		-nx4 matrix
 *
 * This code calculates the inverse kinematics of a manipulator.
 * It calculates the desired end-effector's position and orientation
 * by solving the inverse kinematics, iteratively and concurrently.
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>
// For CUDA runtime routines
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

#define MAX_BODIES 30
#define INPUT_NUM 33

#define NUMBER_OF_BODIES 0
#define JOINT_TYPE 1
#define Q 2
#define DH_PARAMS 3
#define PDES 4
#define ODES 5
#define METHOD 6

// Constant factor for inverse kinematics
#define GAIN 0.1
#define TOLERANCE 0.001
#define MAX_ITERATION 1000

// Global variables to store model details
int nb = 0; // Number of bodies
int jType[MAX_BODIES]; // Joint types
float q[MAX_BODIES]; // Vector of joint variables
float DH_params[MAX_BODIES][4]; // Denavit-Hartenberg Parameters

float T_global[4][4];
float J_global[6][MAX_BODIES];

float pdes[2];
float pdes_[3];
float odes;
float odes_[3];
int method;
int dim;

/*
__global__ void
testing(int *xx, int *yy, int *zz) {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	__shared__ int A[16];
	__shared__ int B[16];

	if(y < 4) {
		A[y] = xx[y];
	}
	if(z < 4) {
		B[z] = yy[z];
	}
	if(y < 4 && z < 4) {
		zz[y + 4 * z] = A[y] + B[z];
	}

	//printf("%d y\n", y);
	//printf("%d x\n", x);
	//printf("%d z\n", z);
}
*/

/**
 * Forward Kinematics implemented in parallel using CUDA GPU threads
 */
__global__ void
forwardKinematics(const int n, int *jType_Input, float *q_Input, float *dh_params_input,
		float *t_output, float *j_output, float *test) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	//int y = blockIdx.y * blockDim.y + threadIdx.y;
	//int z = blockIdx.z * blockDim.z + threadIdx.z;

	// Variables for homogeneous matrix calculations
	__shared__ int jType_cuda[MAX_BODIES];
	__shared__ float q_cuda[MAX_BODIES], dh_params_cuda[MAX_BODIES * 4], J[6 * MAX_BODIES];
	__shared__ float T[16], Ti[MAX_BODIES * 16], res[16]; // 4 x 4 = 16
	__shared__ float a_i[MAX_BODIES], alpha_i[MAX_BODIES], d_i[MAX_BODIES], theta_i[MAX_BODIES];
	__shared__ float ct[MAX_BODIES], st[MAX_BODIES], ca[MAX_BODIES], sa[MAX_BODIES];

	// Variables for Jacobian calculations
	__shared__ float pe[3 * MAX_BODIES], zi1[3 * MAX_BODIES], pi1[3 * MAX_BODIES],
		Jp[3 * MAX_BODIES], Jw[3 * MAX_BODIES], p[3 * MAX_BODIES];
	__shared__ float Ti1[MAX_BODIES * 16]; // 4 x 4 = 16

	// To initialise all variables to zero
	if(x >= 300 && x < 316) { // size 16
		T[x - 300] = 0;
	}
	/*
	if(x >= 316 && x < 332) { // size 16
		res[x - 316] = 0;
	}
	if(x >= 332 && x < 348) { // size 16
		res2[x - 332] = 0;
	}
	*/
	if(x >= 348 && x < 364) { // size 16
		Ti1[x - 348] = 0;
	}
	if(x >= 364 && x < 367) { // size 3
		pi1[x - 364] = 0;
	}
	if(x >= 367 && x < 370) { // size 3
		zi1[x - 367] = 0;
	}
	if(x >= 370 && x < 370 + (n * 6)) { // size n x 6
		J[x] = 0;
	}

	// Copy variables for joint Type and Q
	if(x < n) {
		jType_cuda[x] = jType_Input[x];
	}
	if(x >= n && x < 2 * n) {
		q_cuda[x - n] = q_Input[x - n];
	}

	// Copy dh params
	if(x >= 2 * n && x < n * 6) {
		dh_params_cuda[x - 2 * n] = dh_params_input[x - 2 * n];
	}

	__syncthreads();

	// Assign T diagonal
	if(x == 511) {
		T[0] = 1;
		T[5] = 1;
		T[10] = 1;
		T[15] = 1;
	}

	// Assign Ti1 diagonal
	if(x == 510) {
		Ti1[0] = 1;
		Ti1[5] = 1;
		Ti1[10] = 1;
		Ti1[15] = 1;
	}

	// Assign last element of zi1 to 1
	if(x == 509) {
		zi1[2] = 1;
	}

	// Assign DH params for prismatic and revolute joints
	if(x < n) {
		if(jType_cuda[x] == 1) { // Prismatic
			dh_params_cuda[x * 4 + 2] = q_cuda[x];
		}
		else { // Resolute
			dh_params_cuda[x * 4 + 3] = q_cuda[x];
		}
	}

	__syncthreads();

	// Assign a_i, alpha_i, d_i, theta_i
	if(x < n) {
		a_i[x] = dh_params_cuda[x * 4];
	}
	if(x >= n && x < 2 * n) {
		alpha_i[x % n] = dh_params_cuda[(x - n) * 4 + 1];
	}
	if(x >= 2 * n && x < 3 * n) {
		d_i[x % n] = dh_params_cuda[(x - 2 * n) * 4 + 2];
	}
	if(x >= 3 * n && x < 4 * n) {
		theta_i[x % n] = dh_params_cuda[(x - 3 * n) * 4 + 3];
	}

	__syncthreads();

	// Assign ct, st, ca, sa
	if(x < n) {
		ct[x] = cos(theta_i[x]);
	}
	if(x >= n && x < 2 * n) {
		st[x % n] = sin(theta_i[x % n]);
	}
	if(x >= 2 * n && x < 3 * n) {
		ca[x % n] = cos(alpha_i[x % n]);
	}
	if(x >= 3 * n && x < 4 * n) {
		sa[x % n] = sin(alpha_i[x % n]);
	}

	__syncthreads();

	// Assign matrix Ti
	if(x < n) {
		Ti[x * 16 + 0] = ct[x]; 			Ti[x * 16 + 1] = -st[x] * ca[x];				Ti[x * 16 + 2] = st[x] * sa[x];					Ti[x * 16 + 3] = a_i[x] * ct[x];
	}
	if(x >= n && x < 2 * n) {
		Ti[(x % n) * 16 + 4] = st[x % n];	Ti[(x % n) * 16 + 5] = ct[x % n] * ca[x % n];	Ti[x % n * 16 + 6] = -ct[x % n] * sa[x % n];	Ti[x % n * 16 + 7] = a_i[x % n] * st[x % n];
	}
	if(x >= 2 * n && x < 3 * n) {
		Ti[(x % n) * 16 + 8] = 0;			Ti[(x % n) * 16 + 9] = sa[x % n];				Ti[(x % n) * 16 + 10] = ca[x % n];				Ti[(x % n) * 16 + 11] = d_i[x % n];
	}
	if(x >= 3 * n && x < 4 * n) {
		Ti[(x % n) * 16 + 12] = 0;			Ti[(x % n) * 16 + 13] = 0;						Ti[(x % n) * 16 + 14] = 0;						Ti[(x % n) * 16 + 15] = 1;
	}

	__syncthreads();

	// Matrix multiplication T = T * Ti[i]
	for(int i = 0; i < n; i++) {
		if(x < 16) { // 1 thread per 1 value in 4 x 4 matrix
			res[x] = 0;

			for(int a = 0; a < 4; a++) {
				res[x] += T[(x / 4) * 4 + a] * Ti[i * 16 + x % 4 + a * 4];
			}

			// Copy matrix res to T
			T[x] = res[x];
		}

		__syncthreads();
	}


	// Jacobian calculations

	// Initialise pe
	if(x >= 300 && x < 303) {
		pe[x - 300] = T[(x - 300) * 4 + 3];
	}

	// Matrix multiplication Ti1 = Ti1 * Ti
	for(int i = 1; i < n; i++) {
		if(x < 16) {
			res[x] = 0;

			for(int a = 0; a < 4; a++) {
				res[x] += Ti1[(i - 1) * 16 + (x / 4) * 4 + a] * Ti[(i - 1) * 16 + x % 4 + a * 4];
			}

			// Copy res to Ti1
			Ti1[i * 16 + x] = res[x];
		}

		__syncthreads();
	}

	// Assign zi1
	if(x >= n && x < 3 * n) {
		zi1[x] = Ti1[(x / n) * 16 + (x % n) * 4 + 2];
	}

	// Assign pi1
	if(x >= n * 3 && x < 5 * n) {
		pi1[x - 2 * n] = Ti1[((x - 2 * n) / n) * 16 + (x % n) * 4 + 3];
	}

	__syncthreads();

	// Assign p, Jp, and Jw based on joint types
	if(x < 3 * n) {
		if(jType_cuda[x / 3] == 1) { // Prismatic
			Jp[x] = zi1[x];
			Jw[x] = 0;
		}
		else { // Revolute
			p[x] = pe[x % 3] - pi1[x];
			Jw[x] = zi1[x];

			// Cross product
			if(x % 3 == 0) {
				Jp[x] = zi1[x + 1] * p[x + 2] - zi1[x + 2] * p[x + 1];
			}
			if(x % 3 == 1) {
				Jp[x] = zi1[x + 1] * p[x - 1] - zi1[x - 1] * p[x + 1];
			}
			if(x % 3 == 2) {
				Jp[x] = zi1[x - 2] * p[x - 1] - zi1[x - 1] * p[x - 2];
			}
		}
	}

	__syncthreads();

	// Assign J
	if(x < 3 * n) { // top 3 rows
		J[x] = Jp[(x % n) * n + (x / 3)];
	}
	if(x >= 3 * n && x < 6 * n) { // bottom 3 rows
		J[x] = Jw[(x % n) * n + (x / 3) - n];
	}

	// Copy T and J to output
	__syncthreads();
	if(x < 16) {
		t_output[x] = T[x];
	}
	if(x < n * 6) {
		j_output[x] = J[x];
	}
}

/**
 * Forward Kinematics implemented sequentially using CPU processing power.
 */
void forwardKinematicsSequential() {
	float T[4][4] = {{1, 0, 0, 0},
					 {0, 1, 0, 0},
					 {0, 0, 1, 0},
					 {0, 0, 0, 1}};
	float Ti[nb][4][4];
	float J[6][nb];
	float res[4][4]; // To be used for matrix calculations

	int i;
	for(i = 0; i < 6; i++) {
		for(int j = 0; j < nb; j++) {
			J[i][j] = 0;
		}
	}

	for(i = 0; i < nb; i++) {
		if(jType[i] == 1) { // Prismatic
			DH_params[i][2] = q[i];
		}
		else { // Revolute
			DH_params[i][3] = q[i];
		}
	}

	// Computing homogeneous matrix
	for(i = 0; i < nb; i++) {
		float a_i = DH_params[i][0];
		float alpha_i = DH_params[i][1];
		float d_i = DH_params[i][2];
		float theta_i = DH_params[i][3];

		float ct = cos(theta_i);
		float st = sin(theta_i);
		float ca = cos(alpha_i);
		float sa = sin(alpha_i);

		Ti[i][0][0] = ct; Ti[i][0][1] = -st*ca; Ti[i][0][2] = st*sa;  Ti[i][0][3] = a_i*ct;
		Ti[i][1][0] = st; Ti[i][1][1] = ct*ca;  Ti[i][1][2] = -ct*sa; Ti[i][1][3] = a_i*st;
		Ti[i][2][0] = 0;  Ti[i][2][1] = sa;     Ti[i][2][2] = ca;     Ti[i][2][3] = d_i;
		Ti[i][3][0] = 0;  Ti[i][3][1] = 0;      Ti[i][3][2] = 0;      Ti[i][3][3] = 1;

		//T = T * Ti[i]
		//multiplyMatrix(T, Ti[i], T);
		for(int a = 0; a < 4; a++) {
			for(int b = 0; b < 4; b++) {
				res[a][b] = 0;

				for(int c = 0; c < 4; c++) {
					res[a][b] += T[a][c] * Ti[i][c][b];
				}
			}
		}

		// Assign the result to matrix T
		for(int a = 0; a < 4; a++) {
			for(int b = 0; b < 4; b++) {
				T[a][b] = res[a][b];
			}
		}
	}

	// Jacobian Calculations
	float pe[3] = {T[0][3], T[1][3], T[2][3]};
	float zil[3], pil[3], Ti1[4][4], Jp[3], Jw[3], p[3];

	for(i = 0; i < nb; i++) {
		if(i == 0) {
			zil[0] = 0;
			zil[1] = 0;
			zil[2] = 1;

			pil[0] = 0;
			pil[1] = 0;
			pil[2] = 0;

			Ti1[0][0] = 1; Ti1[0][1] = 0; Ti1[0][2] = 0; Ti1[0][3] = 0;
			Ti1[1][0] = 0; Ti1[1][1] = 1; Ti1[1][2] = 0; Ti1[1][3] = 0;
			Ti1[2][0] = 0; Ti1[2][1] = 0; Ti1[2][2] = 1; Ti1[2][3] = 0;
			Ti1[3][0] = 0; Ti1[3][1] = 0; Ti1[3][2] = 0; Ti1[3][3] = 1;
		}
		else {
			//multiplyMatrix(Ti1, Ti[i-1], Ti1);
			for(int a = 0; a < 4; a++) {
				for(int b = 0; b < 4; b++) {
					res[a][b] = 0;

					for(int c = 0; c < 4; c++) {
						res[a][b] += Ti1[a][c] * Ti[i-1][c][b];
					}
				}
			}

			// Assign result to matrix Ti1
			for(int a = 0; a < 4; a++) {
				for(int b = 0; b < 4; b++) {
					Ti1[a][b] = res[a][b];
				}
			}

			zil[0] = Ti1[0][2];
			zil[1] = Ti1[1][2];
			zil[2] = Ti1[2][2];

			pil[0] = Ti1[0][3];
			pil[1] = Ti1[1][3];
			pil[2] = Ti1[2][3];
		}
		if(jType[i] == 1) { // Prismatic
			Jp[0] = zil[0];
			Jp[1] = zil[1];
			Jp[2] = zil[2];

			Jw[0] = 0;
			Jw[1] = 0;
			Jw[2] = 0;
		}
		else { // Revolute
			p[0] = pe[0] - pil[0];
			p[1] = pe[1] - pil[1];
			p[2] = pe[2] - pil[2];

			// Cross product of zil and p
			Jp[0] = zil[1] * p[2] - zil[2] * p[1];
			Jp[1] = zil[2] * p[0] - zil[0] * p[2];
			Jp[2] = zil[0] * p[1] - zil[1] * p[0];

			Jw[0] = zil[0];
			Jw[1] = zil[1];
			Jw[2] = zil[2];
		}

		J[0][i] = Jp[0];
		J[1][i] = Jp[1];
		J[2][i] = Jp[2];
		J[3][i] = Jw[0];
		J[4][i] = Jw[1];
		J[5][i] = Jw[2];
	}

	// Print the results: T and J matrices.
	printf("\n\nResults for sequential algorithm: \n");
	printf("T: \n");
	for(i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			printf("%.4f ", T[i][j]);
		}
		printf("\n");
	}

	printf("\n");
	printf("J: \n");
	for(i = 0; i < 6; i++) {
		for(int j = 0; j < nb; j++) {
			printf("%.4f ", J[i][j]);
		}
		printf("\n");
	}
}

/**
 * Forward Kinematics implemented in parallel using CUDA GPU threads
 */
__global__ void
inverseKinematics() {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
}

/**
 * Inverse Kinematics
 */
void inverseKinematicsSequential() {
	int flag = 0;
	int i = 1;

	// Variables with _ are for spatial robot (where dimension = 3)
	float R[3][3];
	float p[2], p_[3];
	float phi, ori, ori_[3], theta, psi, delta_p[3], delta_p_[6], delta_q[3], delta_q_[6];
	float Jacobian[3][nb], Jacobian_[6][nb], Jacobian_transpose[nb][3], Jacobian_transpose_[nb][6];
	float temp[nb][3], temp_[nb][6];
	float normal;
	float sum;
	float Q_output[nb][1001];

	for(int a = 0; a < nb; a++) {
		Q_output[a][0] = q[a];
	}

	while(flag == 0) {
		// Run forward kinematics
		forwardKinematicsSequential();

		// Copy R
		for(int a = 0; a < 3; a++) {
			for(int b = 0; b < 3; b++) {
				R[a][b] = T_global[a][b];
			}
		}

		if(dim == 2) {
			p[0] = T_global[0][3];
			p[1] = T_global[1][3];

			phi = atan2(R[1][0], R[0][0]);
			ori = phi;

			delta_p[0] = pdes[0] - p[0];
			delta_p[1] = pdes[1] - p[1];
			delta_p[2] = odes - ori;

			// Calculate normal of vector delta_p
			sum = 0;
			for(int a = 0; a < 3; a++) {
				sum += pow(delta_p[a], 2);
			}
			normal = sqrt(sum);
		}
		else {
			p_[0] = T_global[0][3];
			p_[1] = T_global[1][3];
			p_[2] = T_global[2][3];

			phi = atan2(R[1][0], R[0][0]);
			theta = atan2(-R[2][0], sqrt(pow(R[2][1], 2) + pow(R[2][2], 2)));
			psi = atan2(R[2][1], R[2][2]);
			ori_[0] = psi;
			ori_[1] = theta;
			ori_[2] = phi;

			delta_p_[0] = pdes_[0] - p_[0];
			delta_p_[1] = pdes_[1] - p_[1];
			delta_p_[2] = pdes_[2] - p_[2];
			delta_p_[3] = odes_[0] - ori_[0];
			delta_p_[4] = odes_[1] - ori_[1];
			delta_p_[5] = odes_[2] - ori_[2];

			// Calculate normal of vector delta_p
			sum = 0;
			for(int a = 0; a < 6; a++) {
				sum += pow(delta_p_[a], 2);
			}
			normal = sqrt(sum);
		}

		if(normal < TOLERANCE) {
			flag = 1;
		}
		else {
			if(dim == 2) {
				for(int a = 0; a < nb; a++) {
					Jacobian[0][a] = J_global[0][a];
					Jacobian[1][a] = J_global[1][a];
					Jacobian[2][a] = J_global[5][a];
				}

				// Transpose of J
				for(int a = 0; a < nb; a++) {
					for(int b = 0; b < 3; b++) {
						Jacobian_transpose[b][a] = Jacobian[a][b];
						temp[b][a] = GAIN * Jacobian_transpose[b][a];
					}
				}

				if(method == 0) {
					// delta_q = GAIN * transpose(Jacobian) * delta_p
					// nx3 x 3 matrix
					for(int a = 0; a < nb; a++) {
						for(int b = 0; b < 3; b++) {

							for(int c = 0; c < 3; c++) {

							}
						}
					}
				}
				else {
					//delta_q = GAIN * (transpose(Jacobian) / (Jacobian / transpose(Jacobian))) * delta_p
				}
			}
			else {
				for(int a = 0; a < 6; a++) {
					for(int b = 0; b < nb; b++) {
						Jacobian_[a][b] = J_global[a][b];
					}
				}

				// Transpose of J
				for(int a = 0; a < nb; a++) {
					for(int b = 0; b < 6; b++) {
						Jacobian_transpose_[b][a] = Jacobian[a][b];
						temp_[b][a] = GAIN * Jacobian_transpose_[b][a];
					}
				}

				if(method == 0) {
					// delta_q = GAIN * transpose(Jacobian) * delta_p
					// nx6 x 6 matrix
					for(int a = 0; a < nb; a++) {
						for(int b = 0; b < 6; b++) {

							for(int c = 0; c < 6; c++) {

							}
						}
					}
				}
				else {
					//delta_q = GAIN * (transpose(Jacobian) / (Jacobian / transpose(Jacobian))) * delta_p
				}
			}

			i++;
			//Q(:,i) = Q(:, i - 1) + delta_q
			//Q

			if(i > MAX_ITERATION) {
				flag = 2;
			}
		}
	}
}

/**
 * Reads input from a text file. Input is a model of a robot's arm/manipulator.
 * This functions convert the input into global variables.
 */
int readInput() {
	static const char filename[] = "/home/nik/work/cuda/modelInput1.txt";

	FILE *file = fopen(filename, "r");

	if(!file) {
		fprintf(stderr, "Error: failed to open file\n");
		return EXIT_FAILURE;
	}

	char line[128]; // To store input lines

	char strings[INPUT_NUM][128];
	int count = 0;

	printf("Inputs: \n");
	// Copy from text file to array
	while(fgets(line, sizeof(line), file) && count < INPUT_NUM) {
		strcpy(strings[count], line);
		printf("%s", line);
		count++;
	}

	// Assign to global variables
	nb = atoi(strings[NUMBER_OF_BODIES]);

	int index1 = 0, c1, bytesread1;
	int index2 = 0, bytesread2;
	float c2;
	char* input1 = strings[JOINT_TYPE];
	char* input2 = strings[Q];

	// Assign joint type
	while(sscanf(input1, "%d%n", &c1, &bytesread1) > 0) {
		jType[index1++] = c1;
		input1 += bytesread1;
	}

	// Assign q (vector of joint variables)
	while(sscanf(input2, "%f%n", &c2, &bytesread2) > 0) {
		q[index2++] = c2;
		input2 += bytesread2;
	}

	// Assign DH parameters
	for(int i = 0; i < nb; i++) {
		int index4 = 0, bytesread4;
		float c4;
		char* input4 = strings[i + 3];

		while(sscanf(input4, "%f%n", &c4, &bytesread4) > 0) {
			DH_params[i][index4++] = c4;
			input4 += bytesread4;
		}
	}

	// Print variables/model
	printf("\n");
	printf("Number of bodies: %d\n", nb);
	printf("Joint types: ");
	for(int i = 0; i < nb; i++) {
		printf("%d ", jType[i]);
	}
	printf("\nQ: ");
	for(int i = 0; i < nb; i++) {
		printf("%.4f ", q[i]);
	}

	printf("\nDH Params:\n");
	for(int i = 0; i < nb; i++) {
		for(int j = 0; j < 4; j++) {
			printf("%.2f ", DH_params[i][j]);
		}
		printf("\n");
	}

	fclose(file);

	return 0;
}

/**
 * Assign to global variables
 */
int assign(char **p) {


	return 0;
}

/**
 * Check the consistencies of the input file.
 */
int checkInput() {
	// check for number of bodies and length of variables


	return 0;
}

/**
 * Square matrix multiplication
 */
void multiplyMatrix(int mat1[][4], int mat2[][4], int res[][4]) {
	int i, j, k;

	for(i = 0; i < 4; i++) {
		for(j = 0; j < 4; j++) {
			res[i][j] = 0;
			for(k = 0; k < 4; k++) {
				res[i][j] += mat1[i][k] * mat2[k][j];
			}
		}
	}
}

int main() {
	cudaError_t err = cudaSuccess;

	// Check if read input runs successfully
	if(readInput() != 0) {
		fprintf(stderr, "Failed to run readInput() function\n");
		return EXIT_FAILURE;
	}

	// Check for input. Exits the system if inconsistency is found
	if(checkInput() != 0) {
		fprintf(stderr, "Failed to run checkInput() function\n");
		return EXIT_FAILURE;
	}

	// Allocate host variables
	size_t size_int = nb * sizeof(int);
	size_t size_float = nb * sizeof(float);
	size_t size_dh = sizeof(float) * size_t(nb * 4);
	size_t size_T = sizeof(float) * size_t(4 * 4);
	size_t size_J = sizeof(float) * size_t(nb * 6);

	int *host_jType = (int *) malloc(size_int);
	float *host_q = (float *) malloc(size_float);

	// For output
	float *host_T_test = (float *) malloc(size_T);
	float *host_J_test = (float *) malloc(size_J);
	float *host_Test_test = (float *) malloc(size_dh);

	// Check if host variables are successfully allocated
	if(host_q == NULL || host_jType == NULL || host_T_test == NULL ||
			host_J_test == NULL) {
		fprintf(stderr, "Failed to allocate host variables\n");
		exit(EXIT_FAILURE);
	}

	// Copy host variables
	for(int i = 0; i < nb; i++) {
		host_jType[i] = jType[i];
		host_q[i] = q[i];
	}

	// Declare device variables
	int *device_jType = NULL;
	float *device_q = NULL;
	float *device_dh_params = NULL;
	float *device_T = NULL;
	float *device_J = NULL;
	float *device_test = NULL;

	// Allocate device variables
	err = cudaMalloc((void **) &device_jType, size_int);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated device joint types (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &device_q, size_float);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated device q (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &device_dh_params, size_dh);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated device DH parameters (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &device_T, size_T);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device T variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &device_J, size_J);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device J variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &device_test, size_dh);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device test variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	// Copy host variables to device variables
	err = cudaMemcpy(device_jType, host_jType, size_int, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy joint types (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(device_q, host_q, size_float, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy q (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(device_dh_params, DH_params, size_dh, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy DH parameters (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	// CUDA configurations
	int numblocks = 0;
	dim3 blocksPerGrid(512, 1, 1);
	dim3 threadsPerBlock(1024, 1, 1);

	// Code for timing of CUDA kernel function
	float parallel_time;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	forwardKinematics<<<1, 512>>>(nb, device_jType, device_q, device_dh_params,
			device_T, device_J, device_test);
	cudaDeviceSynchronize();

	// End of timing for CUDA
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&parallel_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("\nParallel algorithm time taken: %.4f\n", parallel_time);


	// Copy the device result variables to host variables

	err = cudaMemcpy(host_T_test, device_T, size_T, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy output T to host variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(host_J_test, device_J, size_J, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy output J to host variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(host_Test_test, device_test, size_dh, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy output J to host variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	/*
	printf("\n\n\n ------- Testing variables -------\n\n\n");
	for(int i = 0; i < nb * 4; i++) {
		printf("%.4f ", host_Test_test[i]);
	}
	printf("\n\n\n ------- End of testing variables -------\n\n\n");
	*/
	printf("\nResults for parallel algorithm: \n");
	printf("T: \n");
	for(int i = 0; i < 16; i++) {
		if(i % 4 == 0 && i != 0) {
			printf("\n");
		}
		printf("%.4f ", host_T_test[i]);
	}
	printf("\n");

	printf("\nJ:\n");
	for(int i = 0; i < nb * 6; i++) {
		if(i % nb == 0 && i > 0) {
			printf("\n");
		}
		printf("%.4f ", host_J_test[i]);
	}

	// free device variables
	err = cudaFree(device_jType);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device jType (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_q);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device q (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_dh_params);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device DH parameters (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_T);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device T (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_J);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device J (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Code for timing sequential algorithm
	float sequential_time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Run sequential forward kinematics
	forwardKinematicsSequential();

	//End of timing for sequential algorithm
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&sequential_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("\nSequential algorithm time taken: %.4f", sequential_time);


	// Calculate performance difference
	float difference = parallel_time / sequential_time;
	printf("\nRatio of parallel time to sequential time: %.3f", difference);

	// Free host variables
	free(host_jType);
	free(host_q);
	free(host_T_test);
	free(host_J_test);



	/*
	// ------------------------- Start Code for testing -------------------------

	size_t test_size = 1024 * sizeof(int);
	int *d_test_x = NULL;
	int *d_test_y = NULL;
	int *d_test_z = NULL;

	int *h_test_x = (int *) malloc(test_size);
	int *h_test_y = (int *) malloc(test_size);
	int *h_test_z = (int *) malloc(test_size);

	h_test_x[0] = 100;
	h_test_x[1] = 200;
	h_test_x[2] = 300;
	h_test_x[3] = 400;

	h_test_y[0] = 10;
	h_test_y[1] = 20;
	h_test_y[2] = 30;
	h_test_y[3] = 40;

	err = cudaMalloc((void **) &d_test_x, test_size);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate x test %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &d_test_y, test_size);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate y test %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &d_test_z, test_size);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate z test %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_test_x, h_test_x, test_size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy x test to device %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	err = cudaMemcpy(d_test_y, h_test_y, test_size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy y test to device %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_test_z, h_test_z, test_size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy z test to device %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//int t_numBlocks = (MAX_BODIES + 64 - 1) / 64;
	dim3 t_dimGrid(32, 32, 32);
	dim3 t_dimBlock(64, 64, 64);

	printf("\n\n\nLaunch test\n");
	testing<<<t_dimGrid, 64>>>(d_test_x, d_test_y, d_test_z);

	err = cudaMemcpy(h_test_x, d_test_x, test_size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy x test %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(h_test_y, d_test_y, test_size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy y test %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(h_test_z, d_test_z, test_size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy z test %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("\nTest\n");
	printf("x:\n");
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			printf("%d ", h_test_z[j + 4 * i]);
		}
		printf("\n");
	}


	err = cudaFree(d_test_x);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free x test %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_test_y);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free y test %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_test_z);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free z test %s", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	free(h_test_x);
	free(h_test_y);
	free(h_test_z);


	// ------------------------- End Code for testing -------------------------
	*/


	// Reset device and exit
	err = cudaDeviceReset();
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to reset the device (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("\n\n ---- Done ---- \n\n");


	return 0;
}
