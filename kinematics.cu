/**
 * Author: Nik Zulhilmi Nik Fuaad
 * Computer Science/Software Engineering, University of Birmingham
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
 * 5. PDEs, desired position - (2d float array)
 * 		-2x1 for planar robot
 * 		-3x1 for spatial robot
 * 6. ODEs, desired orientation - (2d float array)
 * 		-scalar for planar
 * 		-3x1 for 3D robot
 * 7. Method (int)
 * 		-value: 0 or 1
 * 8. Dimension (int) : 2 if planar, 3 is spatial
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
#include <time.h>
// For CUDA runtime routines
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

#define MAX_BODIES 80
#define INPUT_NUM 87

#define NUMBER_OF_BODIES 0
#define JOINT_TYPE 1
#define Q 2
#define DH_PARAMS 3
//#define PDES 4
//#define ODES 5
//#define METHOD 6
//#define DIM 7

// Constant factors for inverse kinematics
#define GAIN 0.1
#define TOLERANCE 0.001
#define MAX_ITERATION 10

// Global variables to store model details
int nb = 0; // Number of bodies
int jType[MAX_BODIES]; // Joint types
float q[MAX_BODIES]; // Vector of joint variables
float DH_params[MAX_BODIES][4]; // Denavit-Hartenberg Parameters

float T[4][4];
float J[6][MAX_BODIES];

// Variable specific for inverse kinematics
float pdes[3];
float odes[3];
int method;
int dim; // size of pdes
float Q_output[MAX_BODIES][MAX_ITERATION + 1];

// Variables for checking inputs
int j_length;
int q_length;
int dh_length;
int pdes_length;
int odes_length;

// Variables for CUDA kernel
//__shared__ int jType_cuda[MAX_BODIES];
//__shared__ float q_cuda[MAX_BODIES];
//__shared__ float T_cuda[16];
//__shared__ float J_cuda[MAX_BODIES * 6];
//__shared__ float dh_params_cuda[MAX_BODIES * 4];
//__shared__ float Q_cuda[MAX_BODIES * MAX_ITERATION + 1];

/**
 * Methods for testing purposes.
 */
__device__ void
testing1(float *input) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x < 10) {
		input[x] += 10;
	}
}
__global__ void
testing(int *x_input) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float XY[10];

	if(x < 10) {
		XY[x] = x;
	}

	//testing1
}

/**
 * Forward Kinematics implemented in parallel using CUDA GPU threads
 */
__global__ void
forwardKinematics(const int n, int *jType_Input, float *q_Input, float *dh_params_input,
		float *t_output, float *j_output, int *runtime_first, int *runtime_second) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	//int y = blockIdx.y * blockDim.y + threadIdx.y;
	//int z = blockIdx.z * blockDim.z + threadIdx.z;

	// Variables for homogeneous matrix calculations
	__shared__ float Ti[MAX_BODIES * 16], res[16]; // 4 x 4 = 16
	__shared__ float a_i[MAX_BODIES], alpha_i[MAX_BODIES], d_i[MAX_BODIES], theta_i[MAX_BODIES];
	__shared__ float ct[MAX_BODIES], st[MAX_BODIES], ca[MAX_BODIES], sa[MAX_BODIES];

	// Variables for Jacobian calculations
	__shared__ float pe[3 * MAX_BODIES], zi1[3 * MAX_BODIES], pi1[3 * MAX_BODIES],
		Jp[3 * MAX_BODIES], Jw[3 * MAX_BODIES], p[3 * MAX_BODIES];
	__shared__ float Ti1[MAX_BODIES * 16]; // 4 x 4 = 16
	__shared__ float T_cuda[16], J_cuda[6 * MAX_BODIES];
	__shared__ float jType_cuda[MAX_BODIES], q_cuda[MAX_BODIES], dh_params_cuda[4 * MAX_BODIES];

	// To initialise all variables to zero
	if(x >= 200 && x < 216) { // size 16
		T_cuda[x - 200] = 0;
	}
	/*
	if(x >= 316 && x < 332) { // size 16
		res[x - 316] = 0;
	}
	*/
	if(x >= 248 && x < 264) { // size 16
		Ti1[x - 248] = 0;
	}
	if(x >= 264 && x < 267) { // size 3
		pi1[x - 264] = 0;
	}
	if(x >= 267 && x < 270) { // size 3
		zi1[x - 267] = 0;
	}
	if(x >= 270 && x < 270 + (n * 6)) { // size n x 6
		J_cuda[x - 270] = 0;
	}

	// Start timing here
	clock_t start_time_first = clock64();

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

	// end timing here
	clock_t stop_time_first = clock64();

	__syncthreads();

	// Assign T diagonal
	if(x == 500) {
		T_cuda[0] = 1;
		T_cuda[5] = 1;
		T_cuda[10] = 1;
		T_cuda[15] = 1;
	}

	// Assign Ti1 diagonal
	if(x == 501) {
		Ti1[0] = 1;
		Ti1[5] = 1;
		Ti1[10] = 1;
		Ti1[15] = 1;
	}

	// Assign last element of zi1 to 1
	if(x == 502) {
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
		alpha_i[x % n] = dh_params_cuda[(x % n) * 4 + 1];
	}
	if(x >= 2 * n && x < 3 * n) {
		d_i[x % n] = dh_params_cuda[(x % n) * 4 + 2];
	}
	if(x >= 3 * n && x < 4 * n) {
		theta_i[x % n] = dh_params_cuda[(x % n) * 4 + 3];
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
		Ti[(x % n) * 16 + 4] = st[x % n];	Ti[(x % n) * 16 + 5] = ct[x % n] * ca[x % n];	Ti[(x % n) * 16 + 6] = -ct[x % n] * sa[x % n];	Ti[(x % n) * 16 + 7] = a_i[x % n] * st[x % n];
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
				res[x] += T_cuda[(x / 4) * 4 + a] * Ti[i * 16 + x % 4 + a * 4];
			}

			// Copy matrix res to T
			T_cuda[x] = res[x];
		}

		__syncthreads();
	}

	// Jacobian calculations

	// Initialise pe
	if(x >= 300 && x < 303) {
		pe[x - 300] = T_cuda[(x - 300) * 4 + 3];
	}

	// Matrix multiplication Ti1 = Ti1 * Ti
	for(int i = 1; i < n; i++) {
		if(x < 16) {
			res[x] = 0;

			for(int a = 0; a < 4; a++) {
				res[x] += Ti1[(i - 1) * 16 + (x / 4) * 4 + a] * Ti[(i - 1) * 16 + (x % 4) + a * 4];
			}

			// Copy res to Ti1
			Ti1[i * 16 + x] = res[x];
		}

		__syncthreads();
	}

	// Assign zi1
	if(x >= 3 && x < 3 * n) {
		zi1[x] = Ti1[(x / 3) * 16 + (x % 3) * 4 + 2];
	}

	// Assign pi1
	if(x >= 3 + 3 * n && x < 6 * n) {
		pi1[x - 3 * n] = Ti1[((x - 3 * n) / 3) * 16 + (x % 3) * 4 + 3];
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
		J_cuda[x] = Jp[(x % n) * 3 + (x / n)];
	}
	if(x >= 3 * n && x < 6 * n) { // bottom 3 rows
		J_cuda[x] = Jw[(x % n) * 3 + ((x - 3 * n) / n)];
	}

	// Start timing here
	clock_t start_time_second = clock64();

	// Copy T and J to output
	__syncthreads();
	if(x < 16) {
		t_output[x] = T_cuda[x];
	}
	if(x < n * 6) {
		j_output[x] = J_cuda[x];
	}

	// End timing here
	clock_t stop_time_second = clock64();

	runtime_first[x] = (int)(stop_time_first - start_time_first);
	runtime_second[x] = (int)(stop_time_second - start_time_second);
}

/**
 * Forward Kinematics implemented sequentially using CPU processing power.
 */
void forwardKinematicsSequential() {
	// Initialise T to zeros
	for(int a = 0; a < 4; a++) {
		for(int b = 0; b < 4; b++) {
			T[a][b] = 0;
		}
	}
	// Assign diagonal of T = 1
	T[0][0] = 1; T[1][1] = 1; T[2][2] = 1; T[3][3] = 1;

	float Ti[nb][4][4];
	float res[4][4]; // To be used for matrix calculations

	int i;
	for(i = 0; i < 6; i++) {
		for(int j = 0; j < nb; j++) {
			J[i][j] = 0;
		}
	}

	// Set DH params according to joint types
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

	// Iterate through every single link
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

		// Assign to J
		J[0][i] = Jp[0];
		J[1][i] = Jp[1];
		J[2][i] = Jp[2];
		J[3][i] = Jw[0];
		J[4][i] = Jw[1];
		J[5][i] = Jw[2];
	}
}

/**
 * Find the determinant of a 3x3 or 6x6 matrix.
 * Helper function for the calculation of inverse matrix.
 */
float determinant(float a[6][6], float k) {
	float s = 1, det = 0, b[6][6];
	int i, j, m, n, c;

	if(k == 1) {
		return (a[0][0]);
	}
	else {
		det = 0;
		for(c = 0; c < k; c++) {
			m = 0;
			n = 0;

			for(i = 0; i < k; i++) {
				for(j = 0; j < k; j++) {
					b[i][j] = 0;

					if(i != 0 && j != c) {
						b[m][n] = a[i][j];

						if(n < k - 2) {
							n++;
						}
						else {
							n = 0;
							m++;
						}
					}
				}
			}

			det = det + s * (a[0][c] * determinant(b, k-1));
			s = -1 * s;
		}
	}

	return det;
}

/**
 * Find the transpose of a matrix, to find inverse.
 * Helper function for the calculation of inverse matrix.
 */
void transpose(float num[6][6], float fac[6][6], float r, float res[6][6]) {
	int i, j;
	float b[6][6], d;

	for(i = 0; i < r; i++) {
		for(j = 0; j < r; j++) {
			b[i][j] = fac[j][i];
		}
	}

	d = determinant(num, r);

	for(i = 0; i < r; i++) {
		for(j = 0; j < r; j++) {
			res[i][j] = b[i][j] / d;
		}
	}
}

/**
 * Find the cofactor of a 3x3 or 6x6 matrix
 * Helper function for the calculation of inverse matrix.
 */
void cofactor(float num[6][6], float f, float res[6][6]) {
	float b[6][6], fac[6][6];
	int p, q, m, n, i, j;

	for(q = 0; q < f; q++) {
		for(p = 0; p < f; p++) {
			m = 0;
			n = 0;

			for(i = 0; i < f; i++) {
				for(j = 0; j < f; j++) {
					if(i != q && j != p) {
						b[m][n] = num[i][j];

						if(n < f - 2) {
							n++;
						}
						else {
							n = 0;
							m++;
						}
					}
				}
			}

			fac[q][p] = pow(-1, q + p) * determinant(b, f - 1);
		}
	}

	transpose(num, fac, f, res);
}

/**
 * Test method for inverse matrix
 */
void testMatrix() {
	printf("\n\nTesting matrix\n");

	float m1[3][3] = {{1.0, 3.0, 3.0},
					  {1.0, 4.0, 3.0},
					  {1.0, 3.0, 4.0}};

	float m2[6][6] = {{1.0, 5.0, 6.0, 8.0, 6.0, 5.0},
					  {3.0, 4.0, 6.0, 8.0, 7.0, 4.0},
					  {3.0, 4.0, 6.0, 8.0, 0.0, 1.0},
					  {2.0, 4.0, 8.0, 6.0, 0.0, 5.0},
					  {3.0, 5.0, 8.0, 0.0, 9.0, 9.0},
					  {5.0, 6.0, 6.0, 8.0, 9.0, 0.0}};

	float m3[6][6];
	m3[0][0] = 1.0; m3[0][1] = 3.0; m3[0][2] = 3.0;
	m3[1][0] = 1.0; m3[1][1] = 4.0; m3[1][2] = 3.0;
	m3[2][0] = 1.0; m3[2][1] = 3.0; m3[2][2] = 4.0;

	float res[6][6];

	float d;
	d = determinant(m2, 6);

	if(d == 0) {
		printf("\nDeterminant of the matrix = 0\n");
	}
	else {
		cofactor(m2, 6, res);
	}

	printf("\nResults:\n");
	for(int i = 0; i < 6; i++) {
		for(int j = 0; j < 6; j++) {
			printf("%.2f ", res[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

/**
 * Inverse Kinematics implemented in parallel using CUDA GPU threads
 */
__global__ void
inverseKinematics(const int n, const int method, const int dim, float *pdes_input,
		float *odes_input, float *q_output, int *runtime_first, int *runtime_second,
		float *dh_params_input, float *q_input, int *jType_input) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	//int y = blockIdx.y * blockDim.y + threadIdx.y;
	//int z = blockIdx.z * blockDim.z + threadIdx.z;

	int flag = 0;
	int i = 1;

	// Variables for inverse kinematics
	__shared__ float phi[1], theta[1], psi[1], normal[1], sum[1];
	__shared__ float R[9], p_ik[3], ori[3], delta_p[6], delta_q[MAX_BODIES];
	__shared__ float Jacobian[6 * MAX_BODIES], Jacobian_T[MAX_BODIES * 6];
	__shared__ float temp[6 * 6], inverse[6 * 6];

	// Variables for forward kinematics
	// Variables for homogeneous matrix calculations
	__shared__ float Ti[MAX_BODIES * 16], res[16]; // 4 x 4 = 16
	__shared__ float a_i[MAX_BODIES], alpha_i[MAX_BODIES], d_i[MAX_BODIES], theta_i[MAX_BODIES];
	__shared__ float ct[MAX_BODIES], st[MAX_BODIES], ca[MAX_BODIES], sa[MAX_BODIES];

	// Variables for Jacobian calculations
	__shared__ float pe[3 * MAX_BODIES], zi1[3 * MAX_BODIES], pi1[3 * MAX_BODIES],
		Jp[3 * MAX_BODIES], Jw[3 * MAX_BODIES], p_fk[3 * MAX_BODIES];
	__shared__ float Ti1[MAX_BODIES * 16]; // 4 x 4 = 16

	// External variables
	__shared__ float pdes_cuda[3], odes_cuda[3];
	__shared__ float Q_cuda[MAX_BODIES * MAX_ITERATION + 1];
	__shared__ float T_cuda[16], J_cuda[6 * MAX_BODIES];
	__shared__ float jType_cuda[MAX_BODIES], q_cuda[MAX_BODIES], dh_params_cuda[4 * MAX_BODIES];

	/*
	// For testing
	if(x == 500) {
		pdes_cuda[x - 400] = 0.3818;
	}
	if(x == 501) {
		pdes_cuda[x - 400] = 0.9122;
	}
	if(x == 502) {
		pdes_cuda[x - 400] = 0.4924;
	}
	if(x == 503) {
		odes_cuda[x - 403] = 0.3550;
	}
	if(x == 504) {
		odes_cuda[x - 403] = 0.2810;
	}
	if(x == 505) {
		odes_cuda[x - 403] = 0.9727;
	}
	*/

	// Start timing here
	clock_t start_time_first = clock64();

	// Copy variables
	if(x >= n && x < 2 * n) {
		jType_cuda[x - n] = jType_input[x - n];
	}
	if(x >= 2 * n && x < 3 * n) {
		q_cuda[x - 2 * n] = q_input[x - 2 * n];

		// Assign Q_cuda
		Q_cuda[(x - 2 * n) * (MAX_ITERATION + 1)] = q_cuda[x - 2 * n];
	}

	// Copy dh params
	if(x >= 3 * n && x < n * 7) {
		dh_params_cuda[x - 3 * n] = dh_params_input[x - 3 * n];
	}

	// Copy PDES and ODES
	if(x >= 500 && x < 503) { // size 3
		pdes_cuda[x - 500] = pdes_input[x - 500];
	}
	if(x >= 503 && 506) { // size 3
		odes_cuda[x - 503] = odes_input[x - 503];
	}

	// End timing here
	clock_t stop_time_first = clock64();

	__syncthreads();

	while(flag == 0) {
		// Assign Q
		if(x < n) {
			q_cuda[x] = Q_cuda[(x * (MAX_ITERATION + 1)) + (i - 1)];
		}

		// Initialise delta_q to zeros
		if(x >= n && x < 2 * n) {
			delta_q[x - n] = 0;
		}

		__syncthreads();

		// ----- Run forward kinematics -----

		// To initialise all variables to zero
		if(x >= 200 && x < 216) { // size 16
			T_cuda[x - 200] = 0;
		}
		if(x >= 248 && x < 264) { // size 16
			Ti1[x - 248] = 0;
		}
		if(x >= 264 && x < 267) { // size 3
			pi1[x - 264] = 0;
		}
		if(x >= 267 && x < 270) { // size 3
			zi1[x - 267] = 0;
		}
		if(x >= 270 && x < 270 + (n * 6)) { // size n x 6
			J_cuda[x - 270] = 0;
		}

		__syncthreads();

		// Assign T diagonal
		if(x == 500) {
			T_cuda[0] = 1;
			T_cuda[5] = 1;
			T_cuda[10] = 1;
			T_cuda[15] = 1;
		}

		// Assign Ti1 diagonal
		if(x == 501) {
			Ti1[0] = 1;
			Ti1[5] = 1;
			Ti1[10] = 1;
			Ti1[15] = 1;
		}

		// Assign last element of zi1 to 1
		if(x == 502) {
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
			alpha_i[x % n] = dh_params_cuda[(x % n) * 4 + 1];
		}
		if(x >= 2 * n && x < 3 * n) {
			d_i[x % n] = dh_params_cuda[(x % n) * 4 + 2];
		}
		if(x >= 3 * n && x < 4 * n) {
			theta_i[x % n] = dh_params_cuda[(x % n) * 4 + 3];
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
			Ti[(x % n) * 16 + 4] = st[x % n];	Ti[(x % n) * 16 + 5] = ct[x % n] * ca[x % n];	Ti[(x % n) * 16 + 6] = -ct[x % n] * sa[x % n];	Ti[(x % n) * 16 + 7] = a_i[x % n] * st[x % n];
		}
		if(x >= 2 * n && x < 3 * n) {
			Ti[(x % n) * 16 + 8] = 0;			Ti[(x % n) * 16 + 9] = sa[x % n];				Ti[(x % n) * 16 + 10] = ca[x % n];				Ti[(x % n) * 16 + 11] = d_i[x % n];
		}
		if(x >= 3 * n && x < 4 * n) {
			Ti[(x % n) * 16 + 12] = 0;			Ti[(x % n) * 16 + 13] = 0;						Ti[(x % n) * 16 + 14] = 0;						Ti[(x % n) * 16 + 15] = 1;
		}

		__syncthreads();

		// Matrix multiplication T = T * Ti[i]
		for(int j = 0; j < n; j++) {
			if(x < 16) { // 1 thread per 1 value in 4 x 4 matrix
				res[x] = 0;

				for(int a = 0; a < 4; a++) {
					res[x] += T_cuda[(x / 4) * 4 + a] * Ti[j * 16 + x % 4 + a * 4];
				}

				// Copy matrix res to T
				T_cuda[x] = res[x];
			}

			__syncthreads();
		}

		// Jacobian calculations

		// Initialise pe
		if(x >= 300 && x < 303) {
			pe[x - 300] = T_cuda[(x - 300) * 4 + 3];
		}

		// Matrix multiplication Ti1 = Ti1 * Ti
		for(int j = 1; j < n; j++) {
			if(x < 16) {
				res[x] = 0;

				for(int a = 0; a < 4; a++) {
					res[x] += Ti1[(j - 1) * 16 + (x / 4) * 4 + a] * Ti[(j - 1) * 16 + (x % 4) + a * 4];
				}

				// Copy res to Ti1
				Ti1[j * 16 + x] = res[x];
			}

			__syncthreads();
		}

		// Assign zi1
		if(x >= 3 && x < 3 * n) {
			zi1[x] = Ti1[(x / 3) * 16 + (x % 3) * 4 + 2];
		}

		// Assign pi1
		if(x >= 3 + 3 * n && x < 6 * n) {
			pi1[x - 3 * n] = Ti1[((x - 3 * n) / 3) * 16 + (x % 3) * 4 + 3];
		}

		__syncthreads();

		// Assign p, Jp, and Jw based on joint types
		if(x < 3 * n) {
			if(jType_cuda[x / 3] == 1) { // Prismatic
				Jp[x] = zi1[x];
				Jw[x] = 0;
			}
			else { // Revolute
				p_fk[x] = pe[x % 3] - pi1[x];
				Jw[x] = zi1[x];

				// Cross product
				if(x % 3 == 0) {
					Jp[x] = zi1[x + 1] * p_fk[x + 2] - zi1[x + 2] * p_fk[x + 1];
				}
				if(x % 3 == 1) {
					Jp[x] = zi1[x + 1] * p_fk[x - 1] - zi1[x - 1] * p_fk[x + 1];
				}
				if(x % 3 == 2) {
					Jp[x] = zi1[x - 2] * p_fk[x - 1] - zi1[x - 1] * p_fk[x - 2];
				}
			}
		}

		__syncthreads();

		// Assign J
		if(x < 3 * n) { // top 3 rows
			J_cuda[x] = Jp[(x % n) * 3 + (x / n)];
		}
		if(x >= 3 * n && x < 6 * n) { // bottom 3 rows
			J_cuda[x] = Jw[(x % n) * 3 + ((x - 3 * n) / n)];
		}

		// ----- End of forward kinematics -----


		__syncthreads();

		// Copy R
		if(x >= 400 && x < 409) {
			R[x - 400] = T_cuda[(x - 400) + (x - 400) / 3];
		}
		if(x == 409) {
			sum[0] = 0;
		}
		if(x >= 410 && x < 412) {
			p_ik[x - 410] = T_cuda[(x - 410) * 4 + 3];
		}

		__syncthreads();

		if(x == 0) {
			phi[0] = atan2(R[3], R[0]);
		}
		if(x >= 412 && x < 414) {
			delta_p[x - 412] = pdes_cuda[x - 412] - p_ik[x - 412];
		}

		__syncthreads();

		if(dim == 2) {
			if(x == 1) {
				ori[0] = phi[0];
				delta_p[2] = odes_cuda[0] - ori[0];

				// Calculate normal of vector delta_p
				for(int a = 0; a < 3; a++) {
					sum[0] += pow(delta_p[a], 2);
				}

				normal[0] = sqrt(sum[0]);
			}
		}
		else { // dim == 3
			if(x == 1) {
				p_ik[2] = T_cuda[11];
			}
			if(x == 2) {
				theta[0] = atan2(-R[6], sqrt(pow(R[7], 2) + pow(R[8], 2)));
			}
			if(x == 3) {
				psi[0] = atan2(R[7], R[8]);
			}

			__syncthreads();

			if(x == 0) {
				ori[x] = psi[0];
				delta_p[3] = odes_cuda[x] - ori[x];
			}
			if(x == 1) {
				ori[x] = theta[0];
				delta_p[4] = odes_cuda[x] - ori[x];
			}
			if(x == 2) {
				ori[x] = phi[0];
				delta_p[5] = odes_cuda[x] - ori[x];
			}
			if(x == 3) {
				delta_p[2] = pdes_cuda[2] - p_ik[2];
			}

			__syncthreads();

			// Calculate normal of vector delta_p
			if(x == 0) {
				for(int a = 0; a < 6; a++) {
					sum[0] += pow(delta_p[a], 2);
				}

				normal[0] = sqrt(sum[0]);
			}

		}

		__syncthreads();

		if(normal[0] < TOLERANCE) {
			flag = 1;
		}
		else {
			if(dim == 2) {
				if(x < 2 * n) { // for first two rows
					// Copy Jacobian from J
					Jacobian[x] = J_cuda[x];

					// Transpose of Jacobian
					Jacobian_T[(x % n) * 3 + (x / n)] = Jacobian[x];
				}
				if(x >= 2 * n && x < 3 * n) {
					Jacobian[x] = J_cuda[x + 3 * n];
					Jacobian_T[(x % n) * 3 + (x / n)] = Jacobian[x];
				}
			}
			else { // dim == 3
				// Copy Jacobian from J
				if(x < 6 * n) {
					// Copy Jacobian from J
					Jacobian[x] = J_cuda[x];

					// Transpose of Jacobian
					Jacobian_T[(x % n) * 6 + (x / n)] = Jacobian[x];
				}
			}

			__syncthreads();

			if(method == 0) {
				// delta_q = GAIN * transpose(Jacobian) * delta_p
				// nx3 x 3x1 matrix
				if(dim == 2) {
					if(x < n) {
						for(int a = 0; a < 3; a++) {
							delta_q[x] += GAIN * delta_p[a] * Jacobian_T[x * 3 + a];
						}
					}
				}
				// nx6 x 6x1 matrix
				else { // dim == 3
					if(x < n) {
						for(int a = 0; a < 6; a++) {
							delta_q[x] += GAIN * delta_p[a] * Jacobian_T[x * 6 + a];
						}
					}
				}
			}

			// Pseudo-inverse matrix J^T * (J * J^T) ^ -1
			else { // method == 1
				// delta_q = GAIN * (transpose(Jacobian) / (Jacobian / transpose(Jacobian))) *delta_p
			}

			__syncthreads();

			// Q(:,1) = Q(:,i-1) + delta_q
			if(x < n) {
				Q_cuda[(x * (MAX_ITERATION + 1)) + i] = Q_cuda[(x * (MAX_ITERATION + 1)) + (i - 1)] + delta_q[x];
			}

			i++;
			if(i > MAX_ITERATION) {
				flag = 2;
			}
		}

		__syncthreads();
	}

	// Start timing here
	clock_t start_time_second = clock64();

	// Copy the output
	if(x < n * (MAX_ITERATION + 1)) {
		q_output[x] = Q_cuda[x];
	}

	// End timing here
	clock_t stop_time_second = clock64();

	// Copy times
	runtime_first[x] = (int) (stop_time_first - start_time_first);
	runtime_second[x] = (int) (stop_time_second - start_time_second);
}

/**
 * Inverse Kinematics
 */
void inverseKinematicsSequential() {
	int flag = 0;
	int i = 1;

	// Variables
	float R[3][3];
	float phi, theta, psi;
	float p[3], ori[3], delta_p[6], delta_q[nb];
	float Jacobian[6][nb], Jacobian_T[nb][6];
	float temp[6][6], inverse[6][6], res[nb][6];
	float normal, sum;
	//float Q_output[nb][MAX_ITERATION + 1];

	/*
	// For testing
	pdes[0] = 0.3818; pdes[1] = 0.9122; pdes[2] = 0.4924;
	odes[0] = 0.3550; odes[1] = 0.2810; odes[2] = 0.9727;
	method = 0;
	dim = 2;
	*/

	for(int a = 0; a < nb; a++) {
		Q_output[a][0] = q[a];
	}

	while(flag == 0) {
		// Run forward kinematics
		for(int a = 0; a < nb; a++) {
			q[a] = Q_output[a][i-1];
		}
		forwardKinematicsSequential();

		// Copy R
		for(int a = 0; a < 3; a++) {
			for(int b = 0; b < 3; b++) {
				R[a][b] = T[a][b];
			}
		}

		p[0] = T[0][3];
		p[1] = T[1][3];

		phi = atan2(R[1][0], R[0][0]);

		delta_p[0] = pdes[0] - p[0];
		delta_p[1] = pdes[1] - p[1];

		sum = 0;

		if(dim == 2) {
			ori[0] = phi;

			delta_p[2] = odes[0] - ori[0];

			// Calculate normal of vector delta_p
			for(int a = 0; a < 3; a++) {
				sum += pow(delta_p[a], 2);
			}
		}
		else {
			p[2] = T[2][3];

			theta = atan2(-R[2][0], sqrt( pow(R[2][1], 2) + pow(R[2][2], 2)) );
			psi = atan2(R[2][1], R[2][2]);

			ori[0] = psi;
			ori[1] = theta;
			ori[2] = phi;

			delta_p[2] = pdes[2] - p[2];
			delta_p[3] = odes[0] - ori[0];
			delta_p[4] = odes[1] - ori[1];
			delta_p[5] = odes[2] - ori[2];

			// Calculate normal of vector delta_p
			for(int a = 0; a < 6; a++) {
				sum += pow(delta_p[a], 2);
			}
		}

		normal = sqrt(sum);

		if(normal < TOLERANCE) {
			flag = 1;
		}
		else {
			if(dim == 2) {
				for(int a = 0; a < nb; a++) {
					Jacobian[0][a] = J[0][a];
					Jacobian[1][a] = J[1][a];
					Jacobian[2][a] = J[5][a];
				}

				// Transpose of J
				for(int a = 0; a < 3; a++) {
					for(int b = 0; b < nb; b++) {
						Jacobian_T[b][a] = Jacobian[a][b];
					}
				}
			}
			else { // dim == 3
				// Jacobian = J
				for(int a = 0; a < 6; a++) {
					for(int b = 0; b < nb; b++) {
						Jacobian[a][b] = J[a][b];
					}
				}

				// Transpose of J
				for(int a = 0; a < 6; a++) {
					for(int b = 0; b < nb; b++) {
						Jacobian_T[b][a] = Jacobian[a][b];
					}
				}
			}

			if(method == 0) {
				// delta_q = GAIN * transpose(Jacobian) * delta_p
				// nx3 x 3 matrix
				for(int a = 0; a < nb; a++) {
					delta_q[a] = 0;

					if(dim == 2) {
						for(int b = 0; b < 3; b++) {
							delta_q[a] += GAIN * Jacobian_T[a][b] * delta_p[b];
						}
					}
					else {
						for(int b = 0; b < 6; b++) {
							delta_q[a] += GAIN * Jacobian_T[a][b] * delta_p[b];
						}
					}
				}
			}
			else {
				//delta_q = GAIN * (transpose(Jacobian) / (Jacobian / transpose(Jacobian))) * delta_p

			}

			for(int a = 0; a < nb; a++) {
				Q_output[a][i] = Q_output[a][i - 1] + delta_q[a];
			}
			i++;

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
	static const char filename[] = "/home/nik/cuda-workspace/Kinematics/src/modelInput1.txt";
	static const char filename_lab[] = "/home/students/nzn448/work/cuda/modelInput1.txt";

	FILE *file = fopen(filename, "r");

	if(!file) {
		file = fopen(filename_lab, "r");
		if(!file) {
			fprintf(stderr, "Error: failed to open file\n");
			return EXIT_FAILURE;
		}
	}

	char line[256]; // To store input lines

	char strings[INPUT_NUM][256];
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
	method = atoi(strings[nb + 5]);
	dim = atoi(strings[nb + 6]);

	int index1 = 0, c1, bytesread1; // J
	int index2 = 0, bytesread2; // Q
	int index3; // DH PARAMS
	int index4 = 0, bytesread4; // PDES
	int index5 = 0, bytesread5; // ODES
	float c2, c4, c5;
	char* input1 = strings[JOINT_TYPE]; // J
	char* input2 = strings[Q]; // Q
	char* input4 = strings[nb + 3]; // PDES
	char* input5 = strings[nb + 4]; // ODES

	// Assign joint type
	while(sscanf(input1, "%d%n", &c1, &bytesread1) > 0) {
		jType[index1++] = c1;
		input1 += bytesread1;
	}
	j_length = index1;

	// Assign q (vector of joint variables)
	while(sscanf(input2, "%f%n", &c2, &bytesread2) > 0) {
		q[index2++] = c2;
		input2 += bytesread2;
	}
	q_length = index2;

	// Assign DH parameters
	for(int i = 0; i < nb; i++) {
		index3 = 0;
		int bytesread3;
		float c3;
		char* input3 = strings[i + 3];

		while(sscanf(input3, "%f%n", &c3, &bytesread3) > 0) {
			DH_params[i][index3++] = c3;
			input3 += bytesread3;
		}
	}
	dh_length = index3;

	// Assign PDES
	while(sscanf(input4, "%f%n", &c4, &bytesread4) > 0) {
		pdes[index4++] = c4;
		input4 += bytesread4;
	}
	pdes_length = index4;

	// Assign ODES
	while(sscanf(input5, "%f%n", &c5, &bytesread5) > 0) {
		odes[index5++] = c5;
		input5 += bytesread5;
	}
	odes_length = index5;

	fclose(file);

	return 0;
}

/**
 * Generate random input variables
 */
int generateRandomVariables(int numberOfInputs) {
	if(numberOfInputs > MAX_BODIES) {
		fprintf(stderr, "Number of inputs is bigger than allowed\n");
		return 0;
	}

	nb = numberOfInputs;

	srand(time(NULL));

	// Generate variables for Q, jType, and DH params
	for(int i = 0; i < numberOfInputs; i++) {
		jType[i] = 0;
		q[i] = (float) rand() / (float) RAND_MAX;

		DH_params[i][0] = (float) rand() / (float) RAND_MAX;
		for(int j = 1; j < 4; j++) {
			DH_params[i][j] = 0;
		}
	}

	// Generate variables for odes_ and odes_
	for(int i = 0; i < 3; i++) {
		pdes[i] = (float) rand() / (float) RAND_MAX;
		odes[i] = rand() % 10;
	}

	method = 0;
	dim = (rand() % (4-2)) + 2;

	return 0;
}

/**
 * Print all the global variables
 */
void printVariables() {
	// Print variables/model
	int i;

	printf("\n");
	printf("Number of bodies: %d", nb);
	printf("\nJoint types: ");
	for(i = 0; i < nb; i++) {
		printf("%d ", jType[i]);
	}

	printf("\nQ: ");
	for(i = 0; i < nb; i++) {
		printf("%.4f ", q[i]);
	}

	printf("\nDH Params:");
	for(i = 0; i < nb; i++) {
		printf("\n");

		for(int j = 0; j < 4; j++) {
			printf("%.4f ", DH_params[i][j]);
		}
	}

	printf("\nPDES: ");
	for(i = 0; i < 3; i++) {
		printf("%.4f ", pdes[i]);
	}

	printf("\nODES: ");
	for(i = 0; i < 3; i++) {
		printf("%.4f ", odes[i]);
	}

	printf("\nMethod: %d", method);
	printf("\nDim: %d", dim);
}

/**
 * Method to analyse the results.This method does the following:
 * 1. Print the times taken for parallel and sequential
 * 2. Compare the times taken
 * 3. Check if the results are the same.
 * 4. If the results aren't the same, print the results for each.
 * 5. If the results are the same, print one
 */
void analyseResults(float s_time_fk, float s_time_ik, float p_time_fk, float p_time_ik,
		float s_T[4][4], float s_J[6][MAX_BODIES], float *p_T, float *p_J, int *runtime_first_fk,
		int *runtime_second_fk, float *p_Q, int *runtime_first_ik, int *runtime_second_ik,
		float epsilon, int blocks) {
	printf("\n\n\n  -------  Results for forward kinematics  -------  ");

	int i, j;

	// Print times
	printf("\nTimes taken for FK:");
	printf("\n   Parallel (includes copying variables): %.4fms", p_time_fk);
	printf("\n   Sequential: %.4fms", s_time_fk);

	// Compare times
	if(s_time_fk < p_time_fk) {
		printf("\nSequential is faster");
	}
	else {
		printf("\nParallel is faster");
	}
	printf("\nRatio of parallel to sequential: %.3f", p_time_fk / s_time_fk);

	// Compare T
	int true_equal_t = 0;
	int true_equal_j = 0;
	for(i = 0; i < 4; i++) {
		for(j = 0; j < 4; j++) {
			if(fabs(s_T[i][j] - p_T[i * 4 + j]) > epsilon) {
				true_equal_t++;
			}
		}
	}

	// Compare J
	for(i = 0; i < 6; i++) {
		for(j = 0; j < nb; j++) {
			if(fabs(s_J[i][j] - p_J[i * nb + j]) > epsilon) {
				true_equal_j++;
			}
		}
	}

	// Check if the results are the same
	if(true_equal_j + true_equal_t < 1) { // Results are the same
		printf("\n\nConsistent outputs");
		// Parallel version
		printf("\nT:\n");
		for(i = 0; i < 16; i++) {
			if(i % 4 == 0 && i != 0) {
				printf("\n");
			}
			printf("%.4f ", p_T[i]);
		}

		printf("\nJ:\n");
		for(i = 0; i < nb * 6; i++) {
			if(i % 6 == 0 && i > 0) {
				printf("\n");
			}
			printf("%.4f ", p_J[i]);
		}
	}
	else {
		printf("\n\n - Inconsistent outputs - ");
		if(true_equal_t > 0) { // T is different
			// Parallel version
			printf("\nT (parallel version):\n");
			for(i = 0; i < 16; i++) {
				if(i % 4 == 0 && i != 0) {
					printf("\n");
				}
				printf("%.4f ", p_T[i]);
			}
			printf("\n");
			// Sequential version
			printf("\nT (sequential version):");
			for(i = 0; i < 4; i++) {
				printf("\n");
				for(j = 0; j < 4; j++) {
					printf("%.4f ", s_T[i][j]);
				}
			}
		}
		if(true_equal_j > 0) { // J is different
			// Parallel version
			printf("\nJ (parallel version):\n");
			for(i = 0; i < 6 * nb; i++) {
				if(i % nb == 0 && i > 0) {
					printf("\n");
				}
				printf("%.4f ", p_J[i]);
			}
			printf("\n");
			// Sequential version
			printf("\nJ (sequential version):");
			for(i = 0; i < 6; i++) {
				printf("\n");
				for(j = 0; j < nb; j++) {
					printf("%.4f ", s_J[i][j]);
				}
			}
		}
	}

	// Array for runtimes
	float r_f_fk[blocks];
	float r_s_fk[blocks];
	float max_f_fk = 0, average_f_fk = 0, min_f_fk;
	float max_s_fk = 0, average_s_fk = 0, min_s_fk;

	// Convert clock cycles to time
	for(i = 0; i < blocks; i++) {
		r_f_fk[i] = runtime_first_fk[i] / 1340000000.0f * 1000.0;
		r_s_fk[i] = runtime_second_fk[i] / 1340000000.0f * 1000.0;
	}

	// Find and calculate the max, min, and average
	for(i = 0; i < blocks; i++) {
		if(r_f_fk[i] > max_f_fk) {
			max_f_fk = r_f_fk[i];
		}
		if(r_s_fk[i] > max_s_fk) {
			max_s_fk = r_s_fk[i];
		}
		average_f_fk += r_f_fk[i];
		average_s_fk += r_s_fk[i];
	}

	min_f_fk = max_f_fk;
	min_s_fk = max_s_fk;
	average_f_fk = average_f_fk / blocks;
	average_s_fk = average_s_fk / blocks;

	for(i = 0; i < blocks; i++) {
		if(r_f_fk[i] < min_f_fk) {
			min_f_fk = r_f_fk[i];
		}
		if(r_s_fk[i] < min_s_fk) {
			min_s_fk = r_s_fk[i];
		}
	}

	// Print runtimes
	printf("\n\nRuntimes:");
	printf("\nFor first copying:");
	printf("\nMin: %.6fms Max: %.6fms Average: %.6fms", min_f_fk, max_f_fk, average_f_fk);
	printf("\nFor second copying:");
	printf("\nMin: %.6fms Max: %.6fms Average: %.6fms", min_s_fk, max_s_fk, average_s_fk);
	printf("\nTotal time taken copying: %.5fms", max_f_fk + max_s_fk);
	printf("\n");

	// For testing:
	//printf("\n\n\n%.6f\n%.6f", max_f_fk, max_s_fk);

	// Results for IK
	printf("\n\n  -------  Results for inverse kinematics  -------  ");

	// Print times
	printf("\nTimes taken for IK:");
	printf("\n   Parallel (includes copying variables): %.4fms", p_time_ik);
	printf("\n   Sequential: %.4fms", s_time_ik);

	// Compare times
	if(s_time_ik < p_time_ik) {
		printf("\nSequential is faster");
	}
	else {
		printf("\nParallel is faster");
	}
	printf("\nRatio of parallel to sequential: %.3f", p_time_ik / s_time_ik);

	// Compare Q
	int true_equal_q = 0;
	for(i = 0; i < nb; i++) {
		for(j = 0; j < MAX_ITERATION + 1; j++) {
			if(fabs(Q_output[i][j] - p_Q[i * (MAX_ITERATION + 1) + j]) > epsilon) {
				true_equal_q++;
			}
		}
	}

	// Check if the results are the same
	if(true_equal_q == 0) {
		printf("\n\nConsistent outputs");
		// Parallel version
		printf("\nQ:\n");
		for(i = 0; i < nb * (MAX_ITERATION + 1); i++) {
			if(i % (MAX_ITERATION + 1) == 0 && i > 0) {
				printf("\n");
			}
			printf("%.4f ", p_Q[i]);
		}
	}
	else {
		printf("\n\n - Inconsistent outputs - ");
		// Parallel version
		printf("\nQ (parallel version):\n");
		for(i = 0; i < nb * (MAX_ITERATION + 1); i++) {
			if(i % (MAX_ITERATION + 1) == 0 && i > 0) {
				printf("\n");
			}
			printf("%.4f ", p_Q[i]);
		}
		printf("\n");
		// Sequential version
		printf("\nQ (sequential version):");
		for(i = 0; i < nb; i++) {
			printf("\n");
			for(j = 0; j < (MAX_ITERATION + 1); j++) {
				printf("%.4f ", Q_output[i][j]);
			}
		}
	}

	// Array for runtimes
	float r_f_ik[blocks];
	float r_s_ik[blocks];
	float max_f_ik = 0, average_f_ik = 0, min_f_ik;
	float max_s_ik = 0, average_s_ik = 0, min_s_ik;

	// Convert clock cycles to time
	for(i = 0; i < blocks; i++) {
		r_f_ik[i] = runtime_first_ik[i] / 1340000000.0f * 1000.0;
		r_s_ik[i] = runtime_second_ik[i] / 1340000000.0f * 1000.0;
	}

	// Find and calculate the max, min, and average
	for(i = 0; i < blocks; i++) {
		if(r_f_ik[i] > max_f_ik) {
			max_f_ik = r_f_ik[i];
		}
		if(r_s_ik[i] > max_s_ik) {
			max_s_ik = r_s_ik[i];
		}
		average_f_ik += r_f_ik[i];
		average_s_ik += r_s_ik[i];
	}

	min_f_ik = max_f_ik;
	min_s_ik = max_s_ik;
	average_f_ik = average_f_ik / blocks;
	average_s_ik = average_s_ik / blocks;

	for(i = 0; i < blocks; i++) {
		if(r_f_ik[i] < min_f_ik) {
			min_f_ik = r_f_ik[i];
		}
		if(r_s_ik[i] < min_s_ik) {
			min_s_ik = r_s_ik[i];
		}
	}

	// Print runtimes
	printf("\n\nRuntimes:");
	printf("\nFor first copying:");
	printf("\nMin: %.6fms Max: %.6fms Average: %.6fms", min_f_ik, max_f_ik, average_f_ik);
	printf("\nFor second copying:");
	printf("\nMin: %.6fms Max: %.6fms Average: %.6fms", min_s_ik, max_s_ik, average_s_ik);
	printf("\nTotal time taken copying: %.5fms", max_f_ik + max_s_ik);
	printf("\n");

}

/**
 * To get the clock rate of the device
 */
void getClockRate() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int clockrate = prop.clockRate;
	printf("\n\nClock rate: %d\n\n", clockrate);
}

/**
 * Check the consistencies of the input file.
 */
int checkInput() {
	// Check for number of bodies and length of variables
	if(!(j_length == nb && q_length == nb &&
			(pdes_length == 2 || pdes_length == 3) &&
			(odes_length == 1 || odes_length == 3) &&
			dh_length == 4)) {
		fprintf(stderr, "Inconsistencies in input dimensions\n");
		return EXIT_FAILURE;
	}

	// Check for method
	if(method < 0 || method > 1) {
		fprintf(stderr, "Method is not equal to 0 or 1");
		return EXIT_FAILURE;
	}

	// Check

	return 0;
}

int main() {
	cudaError_t err = cudaSuccess;
	int blocks = 512;

	//testMatrix();

	//getClockRate();

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

	generateRandomVariables(7);

	printVariables();

	// Allocate host variables
	size_t size_int = nb * sizeof(int);
	size_t size_float = nb * sizeof(float);
	size_t size_dh = sizeof(float) * size_t(nb * 4);
	size_t size_T = sizeof(float) * size_t(4 * 4);
	size_t size_J = sizeof(float) * size_t(nb * 6);
	size_t size_runtime = blocks * sizeof(int);
	size_t size_des = 3 * sizeof(float);
	size_t size_q = sizeof(float) * size_t(nb * (MAX_ITERATION + 1));

	// For copying purposes
	int *host_jType = (int *) malloc(size_int);
	float *host_q = (float *) malloc(size_float);
	float *host_pdes = (float *) malloc(size_des);
	float *host_odes = (float *) malloc(size_des);

	// For analysing
	float analyse_T[4][4];
	float analyse_J[6][MAX_BODIES];
	int *host_runtime_first_fk = (int *) malloc(size_runtime);
	int *host_runtime_second_fk = (int *) malloc(size_runtime);
	int *host_runtime_first_ik = (int *) malloc(size_runtime);
	int *host_runtime_second_ik = (int *) malloc(size_runtime);

	// For output
	float *host_T_test = (float *) malloc(size_T);
	float *host_J_test = (float *) malloc(size_J);
	float *host_Test_test = (float *) malloc(size_dh);
	float *host_Q_output = (float *) malloc(size_q);

	// Check if host variables are successfully allocated
	if(host_q == NULL || host_jType == NULL || host_T_test == NULL || host_J_test == NULL
			|| host_runtime_first_fk == NULL || host_runtime_second_fk == NULL || host_Q_output == NULL
			|| host_pdes == NULL || host_odes == NULL || host_J_test == NULL
			|| host_runtime_first_ik == NULL || host_runtime_second_ik == NULL) {
		fprintf(stderr, "Failed to allocate host variables\n");
		exit(EXIT_FAILURE);
	}

	// Copy host variables
	for(int i = 0; i < nb; i++) {
		host_jType[i] = jType[i];
		host_q[i] = q[i];
	}
	for(int i = 0; i < 3; i++) {
		host_pdes[i] = pdes[i];
		host_odes[i] = odes[i];
	}

	// Declare device variables
	int *device_jType = NULL;
	float *device_q = NULL;
	float *device_dh_params = NULL;
	float *device_T = NULL;
	float *device_J = NULL;
	float *device_test = NULL;
	float *device_odes = NULL;
	float *device_pdes = NULL;
	float *device_Q_output = NULL;
	int *device_runtime_first_fk = NULL;
	int *device_runtime_second_fk = NULL;
	int *device_runtime_first_ik = NULL;
	int *device_runtime_second_ik = NULL;

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

	err = cudaMalloc((void **) &device_runtime_first_fk, size_runtime);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device runtime first FK variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &device_runtime_second_fk, size_runtime);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device runtime second FK variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &device_Q_output, size_q);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated device Q output (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &device_pdes, size_des);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated device PDES (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &device_odes, size_des);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated device ODES (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &device_runtime_first_ik, size_runtime);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated device runtime first IK variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **) &device_runtime_second_ik, size_runtime);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocated device runtime second IK variable (error code %s)\n",
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

	err = cudaMemcpy(device_pdes, host_pdes, size_des, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy PDES (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(device_odes, host_odes, size_des, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy ODES (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// CUDA configurations
	//int numblocks = 0;
	dim3 blocksPerGrid(512, 1, 1);
	dim3 threadsPerBlock(1024, 1, 1);

	// Code for timing of CUDA kernel function
	float parallel_time_fk, parallel_time_ik;
	cudaEvent_t start, stop;

	// Run forward kinematics
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	forwardKinematics<<<1, blocks>>>(nb, device_jType, device_q, device_dh_params,
			device_T, device_J, device_runtime_first_fk, device_runtime_second_fk);
	cudaDeviceSynchronize();

	// End of timing for CUDA
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&parallel_time_fk, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

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

	err = cudaMemcpy(host_runtime_first_fk, device_runtime_first_fk, size_runtime, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy output runtime first FK to host variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(host_runtime_second_fk, device_runtime_second_fk, size_runtime, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy output runtime second FK to host variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	// Run inverse kinematics
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	inverseKinematics<<<1, blocks>>>(nb, method, dim, device_pdes, device_odes, device_Q_output,
			device_runtime_first_ik, device_runtime_second_ik, device_dh_params, device_q,
			device_jType);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&parallel_time_ik, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy the device result variable(s) to host variables

	err = cudaMemcpy(host_Q_output, device_Q_output, size_q, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy output Q_output to host variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(host_runtime_first_ik, device_runtime_first_ik, size_runtime, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy output runtime first IK to host variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(host_runtime_second_ik, device_runtime_second_ik, size_runtime, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy output runtime second IK to host variable (error code %s)\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
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

	err = cudaFree(device_test);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device test (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_runtime_first_fk);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device runtime first FK (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_runtime_second_fk);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device runtime second FK (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_Q_output);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device Q output (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_pdes);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device PDES (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_odes);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device ODES (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_runtime_first_ik);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device runtime first IK (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(device_runtime_second_ik);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to free device runtime second IK (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	// Code for timing sequential algorithm
	float sequential_time_fk, sequential_time_ik;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Run sequential forward kinematics
	forwardKinematicsSequential();

	//End of timing for sequential algorithm
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&sequential_time_fk, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy variables for analysing purposes
	// Copy T
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			analyse_T[i][j] = T[i][j];
		}
	}

	// Copy J
	for(int i = 0; i < 6; i++) {
		for(int j = 0; j < nb; j++) {
			analyse_J[i][j] = J[i][j];
		}
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Run inverse kinematics
	inverseKinematicsSequential();

	//End of timing for sequential algorithm
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&sequential_time_ik, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	analyseResults(sequential_time_fk, sequential_time_ik, parallel_time_fk, parallel_time_ik,
			analyse_T, analyse_J, host_T_test, host_J_test, host_runtime_first_fk, host_runtime_second_fk,
			host_Q_output, host_runtime_first_ik, host_runtime_second_ik, 1e-2, blocks);


	// Free host variables
	free(host_jType);
	free(host_q);
	free(host_T_test);
	free(host_J_test);
	free(host_runtime_first_fk);
	free(host_runtime_second_fk);
	free(host_pdes);
	free(host_odes);
	free(host_Q_output);
	free(host_Test_test);
	free(host_runtime_first_ik);
	free(host_runtime_second_ik);

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

	printf("\n\n  -------  Done  ------- \n\n");

	return 0;
}
