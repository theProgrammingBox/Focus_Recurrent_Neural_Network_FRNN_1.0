#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <chrono>

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

void cpuSgemmStridedBatched(
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	float* B, int ColsB, int SizeB,
	float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

int main()
{
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	curandGenerator_t curandHandle;
	curandCreateGenerator(&curandHandle, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandHandle, 0);
	//curandSetPseudoRandomGeneratorSeed(curandHandle, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());

	const float ONE = 1.0f;
	const float ZERO = 0.0f;

	const uint32_t ACTIONS = 5;				// Number of actions the agent can take
	const uint32_t AGENTS = 3;				// Number of agents

	const uint32_t SENSORY_DIMENSION = 3;	// vector length in each memory, stimulus, and focus
	const uint32_t QUERY_DIMENSION = 7;		// vector length in each query
	const uint32_t VALUE_DIMENSION = 5;		// vector length in each value

	const uint32_t MEMORIES = 7;			// Number of memories per agent, a matrix of size MEMORIES x SENSORY_DIMENSION
	const uint32_t STIMULI = 3;				// Number of stimuli per agent, a matrix of size STIMULI x SENSORY_DIMENSION
	const uint32_t FOCUSES = 5;				// Number of focuses per agent, a matrix of size FOCUSES x SENSORY_DIMENSION

	const uint32_t GLOBAL_SENSORY_VECTORS = MEMORIES + STIMULI;								// number of memories and external stimuli
	const uint32_t LOCAL_SENSORY_VECTORS = GLOBAL_SENSORY_VECTORS + FOCUSES;				// number of memories, external stimuli, and focuses
	const uint32_t SENSORY_ATTENTION_PARAMETERS = 2 * QUERY_DIMENSION + VALUE_DIMENSION;	// number of sensory queries, keys, and values
	const uint32_t CONTEXT_ATTENTION_PARAMETERS = QUERY_DIMENSION + SENSORY_DIMENSION;		// number of context keys and values
	const uint32_t ACTION_VECTORS = ACTIONS + 1;											// vector representations for each action and a vector representing action in general

	const uint32_t INITIAL_SENSORY_MATRIX_SIZE = SENSORY_DIMENSION * LOCAL_SENSORY_VECTORS;						// size of initial memory, stimulus, and focus bias matrix
	const uint32_t SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZE = SENSORY_ATTENTION_PARAMETERS * SENSORY_DIMENSION;	// size of sensory query, key, and value weights
	const uint32_t CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZE = CONTEXT_ATTENTION_PARAMETERS * VALUE_DIMENSION;		// size of context key and value weights
	const uint32_t ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZE = ACTION_VECTORS * SENSORY_DIMENSION;				// size of action weights, a vector representations for each action and a vector representing action in general

	const uint32_t SENSORY_MATRIX_SIZE = INITIAL_SENSORY_MATRIX_SIZE;												// size of memory, stimulus, and focus matrix
	const uint32_t SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE = SENSORY_ATTENTION_PARAMETERS * LOCAL_SENSORY_VECTORS;	// size of sensory query, key, and value matrix
	const uint32_t SENSORY_ATTENTION_SCORE_MATRIX_SIZE = LOCAL_SENSORY_VECTORS * FOCUSES;							// size of sensory attention score matrix
	const uint32_t CONTEXT_MATRIX_SIZE = VALUE_DIMENSION * FOCUSES;													// size of context matrix
	const uint32_t CONTEXT_ATTENTION_PARAMETER_MATRIX_SIZE = CONTEXT_ATTENTION_PARAMETERS * FOCUSES;				// size of context key and value matrix
	const uint32_t CONTEXT_ATTENTION_SCORE_MATRIX_SIZE = FOCUSES * LOCAL_SENSORY_VECTORS;							// size of context attention score matrix
	const uint32_t SELF_UPDATE_MATRIX = SENSORY_DIMENSION * LOCAL_SENSORY_VECTORS;									// size of self update matrix
	const uint32_t ACTION_REPRESENTAION_SCORE_MATRIX_SIZE = ACTION_VECTORS * GLOBAL_SENSORY_VECTORS;				// size of action matrix
	const uint32_t ACTION_VECTOR_SIZE = ACTIONS;																	// size of action probability output

	const uint32_t STATIC_PARAMETERS = INITIAL_SENSORY_MATRIX_SIZE + SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZE + CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZE + ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZE;
	const uint32_t DYNAMIC_PARAMETERS = SENSORY_MATRIX_SIZE + SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE + SENSORY_ATTENTION_SCORE_MATRIX_SIZE + CONTEXT_MATRIX_SIZE + CONTEXT_ATTENTION_PARAMETER_MATRIX_SIZE + CONTEXT_ATTENTION_SCORE_MATRIX_SIZE + SELF_UPDATE_MATRIX + ACTION_REPRESENTAION_SCORE_MATRIX_SIZE + ACTION_VECTOR_SIZE;

	float* workspaceGPU;
	cudaMalloc(&workspaceGPU, STATIC_PARAMETERS + DYNAMIC_PARAMETERS * AGENTS * sizeof(float));	// Allocate memory for static parameters shared by all agents and dynamic parameters are unique to each agent based on their experiences

	float* initialSensoryMatrixGPU = workspaceGPU;																			// Initial memory, stimulus, and focus bias matrix
	float* sensoryAttentionWeightsMatrixGPU = initialSensoryMatrixGPU + INITIAL_SENSORY_MATRIX_SIZE;						// sensory query, key, and value attention weights matrix
	float* contextAttentionWeightsMatrixGPU = sensoryAttentionWeightsMatrixGPU + SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZE;		// context key and value attention weights matrix
	float* actionRepresentationWeightsMatrixGPU = contextAttentionWeightsMatrixGPU + CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZE;	// action representations matrix, (you need to rethink how weights function to understand my naming convention)

	float* sensoryMatrixGPU = actionRepresentationWeightsMatrixGPU + ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZE;				// sensory matrix, contains memories, stimuli, and focuses
	float* sensoryAttentionParameterMatrixGPU = sensoryMatrixGPU + SENSORY_MATRIX_SIZE;										// sensory query, key, and value matrix
	float* sensoryAttentionScoreMatrixGPU = sensoryAttentionParameterMatrixGPU + SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE;	// the attention score matrix of the focus queries on the sensory matrix
	float* contextMatrixGPU = sensoryAttentionScoreMatrixGPU + SENSORY_ATTENTION_SCORE_MATRIX_SIZE;							// context matrix, sum of values based on the focus's attention score
	float* contextAttentionParameterMatrixGPU = contextMatrixGPU + CONTEXT_MATRIX_SIZE;										// using the context gained through focusing on your sensory data compounded over time, make a context key and value matrix to update your sensory data
	float* contextAttentionScoreMatrixGPU = contextAttentionParameterMatrixGPU + CONTEXT_ATTENTION_PARAMETER_MATRIX_SIZE;	// the attention score matrix of the sensory queries matrix on the context keys, kinda think of it like a what should I forget computation
	float* selfUpdateMatrixGPU = contextAttentionScoreMatrixGPU + CONTEXT_ATTENTION_SCORE_MATRIX_SIZE;						// the matrix of how much to update your sensory data based on the sum of the context values based on the context's attention score
	float* actionRepresentationScoreMatrixGPU = selfUpdateMatrixGPU + SELF_UPDATE_MATRIX;									// same as attention, but the keys are static and they represent actions, (its just a regular matrix multiplication, through different perspective)
	float* actionVectorGPU = actionRepresentationScoreMatrixGPU + ACTION_REPRESENTAION_SCORE_MATRIX_SIZE;					// use the sum of if each vector's similarity to each action representation times their similarity to an action in general to get the probability of each action, after a softmax

	float* focusQueriesMatrixGPU = sensoryMatrixGPU + SENSORY_ATTENTION_PARAMETERS * GLOBAL_SENSORY_VECTORS;	// the focus queries matrix
	float* sensoryKeysMatrixGPU = sensoryMatrixGPU + QUERY_DIMENSION;											// the sensory keys matrix
	float* sensoryValuesMatrixGPU = sensoryKeysMatrixGPU + QUERY_DIMENSION;										// the sensory values matrix
	float* contextValuesMatrixGPU = contextMatrixGPU + QUERY_DIMENSION;											// the context values matrix

	curandGenerateNormal(curandHandle, workspaceGPU, STATIC_PARAMETERS + (STATIC_PARAMETERS & 1), 0.0f, 0.4f);	// initialize the initial sensory bias, sensory query, key, and value, context key and value, and action representations, needs to fill in an even number of floats, and needs to be on an even address(I think, just use the address given by cudaMalloc + even number)

	/*// print the initial matrix
	float* initialSensoryMatrixCPU = new float[INITIAL_SENSORY_MATRIX_SIZE];
	cudaMemcpy(initialSensoryMatrixCPU, initialSensoryMatrixGPU, INITIAL_SENSORY_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Agents Initial Sensory Matrix:\n";
	for (int i = 0; i < LOCAL_SENSORY_VECTORS; i++) {
		for (int j = 0; j < SENSORY_DIMENSION; j++)
			std::cout << initialSensoryMatrixCPU[i * SENSORY_DIMENSION + j] << " ";
		cout << "\n";
	}
	cout << "\n";
	delete[] initialSensoryMatrixCPU;

	// print the sensory attention weights matrix
	float* sensoryAttentionWeightsMatrixCPU = new float[SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZE];
	cudaMemcpy(sensoryAttentionWeightsMatrixCPU, sensoryAttentionWeightsMatrixGPU, SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Agents Sensory Attention Weights Matrix:\n";
	for (int i = 0; i < SENSORY_DIMENSION; i++) {
		for (int j = 0; j < SENSORY_ATTENTION_PARAMETERS; j++)
			std::cout << sensoryAttentionWeightsMatrixCPU[i * SENSORY_ATTENTION_PARAMETERS + j] << " ";
		cout << "\n";
	}
	cout << "\n";
	delete[] sensoryAttentionWeightsMatrixCPU;

	// print the context attention weights matrix
	float* contextAttentionWeightsMatrixCPU = new float[CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZE];
	cudaMemcpy(contextAttentionWeightsMatrixCPU, contextAttentionWeightsMatrixGPU, CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Agents Context Attention Weights Matrix:\n";
	for (int i = 0; i < VALUE_DIMENSION; i++) {
		for (int j = 0; j < CONTEXT_ATTENTION_PARAMETERS; j++)
			std::cout << contextAttentionWeightsMatrixCPU[i * CONTEXT_ATTENTION_PARAMETERS + j] << " ";
		cout << "\n";
	}
	cout << "\n";
	delete[] contextAttentionWeightsMatrixCPU;

	// print the action representation weights matrix
	float* actionRepresentationWeightsMatrixCPU = new float[ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZE];
	cudaMemcpy(actionRepresentationWeightsMatrixCPU, actionRepresentationWeightsMatrixGPU, ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Agents Action Representation Weights Matrix:\n";
	for (int i = 0; i < SENSORY_DIMENSION; i++) {
		for (int j = 0; j < ACTION_VECTORS; j++)
			std::cout << actionRepresentationWeightsMatrixCPU[i * ACTION_VECTORS + j] << " ";
		cout << "\n";
	}
	cout << "\n";
	delete[] actionRepresentationWeightsMatrixCPU;*/

	// copy initial sensory matrix to each agent's sensory matrix
	for (int agent = AGENTS; agent--;)
		cudaMemcpy(sensoryMatrixGPU + agent * DYNAMIC_PARAMETERS, initialSensoryMatrixGPU, INITIAL_SENSORY_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToDevice);

	/*// print the agents' sensory matrix
	float* sensoryMatrixCPU = new float[SENSORY_MATRIX_SIZE];
	for (int agent = AGENTS; agent--;) {
		cudaMemcpy(sensoryMatrixCPU, sensoryMatrixGPU + agent * DYNAMIC_PARAMETERS, SENSORY_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Sensory Matrix:\n";
		for (int i = 0; i < LOCAL_SENSORY_VECTORS; i++) {
			for (int j = 0; j < SENSORY_DIMENSION; j++)
				std::cout << sensoryMatrixCPU[i * SENSORY_DIMENSION + j] << " ";
			cout << "\n";
		}
		cout << "\n";
	}
	delete[] sensoryMatrixCPU;*/

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		SENSORY_ATTENTION_PARAMETERS, LOCAL_SENSORY_VECTORS, SENSORY_DIMENSION,
		&ONE,
		sensoryAttentionWeightsMatrixGPU, SENSORY_ATTENTION_PARAMETERS, ZERO,
		sensoryMatrixGPU, SENSORY_DIMENSION, DYNAMIC_PARAMETERS,
		&ZERO,
		sensoryAttentionParameterMatrixGPU, SENSORY_ATTENTION_PARAMETERS, DYNAMIC_PARAMETERS,
		AGENTS);

	// print the sensory attention parameters
	float* sensoryAttentionParameterMatrixCPU = new float[SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE];
	for (int agent = AGENTS; agent--;) {
		cudaMemcpy(sensoryAttentionParameterMatrixCPU, sensoryAttentionParameterMatrixGPU + agent * DYNAMIC_PARAMETERS, SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Sensory Attention Parameter Matrix:\n";
		for (int i = 0; i < LOCAL_SENSORY_VECTORS; i++) {
			for (int j = 0; j < SENSORY_ATTENTION_PARAMETERS; j++)
				std::cout << sensoryAttentionParameterMatrixCPU[i * SENSORY_ATTENTION_PARAMETERS + j] << " ";
			cout << "\n";
		}
		cout << "\n";
	}
	delete[] sensoryAttentionParameterMatrixCPU;/**/

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		LOCAL_SENSORY_VECTORS, FOCUSES, QUERY_DIMENSION,
		&ONE,
		sensoryKeysMatrixGPU, SENSORY_ATTENTION_PARAMETERS, DYNAMIC_PARAMETERS,
		focusQueriesMatrixGPU, SENSORY_ATTENTION_PARAMETERS, DYNAMIC_PARAMETERS,
		&ZERO,
		sensoryAttentionScoreMatrixGPU, LOCAL_SENSORY_VECTORS, DYNAMIC_PARAMETERS,
		AGENTS);

	// print the focus queries
	sensoryAttentionParameterMatrixCPU = new float[SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE];
	float* focusQueriesMatrixCPU = sensoryAttentionParameterMatrixCPU + SENSORY_ATTENTION_PARAMETERS * GLOBAL_SENSORY_VECTORS;
	for (int agent = AGENTS; agent--;) {
		cudaMemcpy(sensoryAttentionParameterMatrixCPU, sensoryAttentionParameterMatrixGPU + agent * DYNAMIC_PARAMETERS, SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Focus Queries:\n";
		for (int i = 0; i < FOCUSES; i++) {
			for (int j = 0; j < QUERY_DIMENSION; j++)
				std::cout << focusQueriesMatrixCPU[i * SENSORY_ATTENTION_PARAMETERS + j] << " ";
			cout << "\n";
		}
		cout << "\n";
	}
	delete[] sensoryAttentionParameterMatrixCPU;

	// print the keys
	sensoryAttentionParameterMatrixCPU = new float[SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE];
	float* sensoryKeysMatrixCPU = sensoryAttentionParameterMatrixCPU + QUERY_DIMENSION;
	for (int agent = AGENTS; agent--;) {
		cudaMemcpy(sensoryAttentionParameterMatrixCPU, sensoryAttentionParameterMatrixGPU + agent * DYNAMIC_PARAMETERS, SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Sensory Keys:\n";
		for (int i = 0; i < LOCAL_SENSORY_VECTORS; i++) {
			for (int j = 0; j < QUERY_DIMENSION; j++)
				std::cout << sensoryKeysMatrixCPU[i * SENSORY_ATTENTION_PARAMETERS + j] << " ";
			cout << "\n";
		}
		cout << "\n";
	}
	delete[] sensoryAttentionParameterMatrixCPU;

	// print the sensory attention scores
	float* sensoryAttentionScoreMatrixCPU = new float[SENSORY_ATTENTION_SCORE_MATRIX_SIZE];
	for (int agent = AGENTS; agent--;) {
		cudaMemcpy(sensoryAttentionScoreMatrixCPU, sensoryAttentionScoreMatrixGPU + agent * DYNAMIC_PARAMETERS, SENSORY_ATTENTION_SCORE_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Sensory Attention Score Matrix:\n";
		for (int i = 0; i < FOCUSES; i++) {
			for (int j = 0; j < LOCAL_SENSORY_VECTORS; j++)
				std::cout << sensoryAttentionScoreMatrixCPU[i * LOCAL_SENSORY_VECTORS + j] << " ";
			cout << "\n";
		}
		cout << "\n";
	}
	delete[] sensoryAttentionScoreMatrixCPU;/**/

	// use cpuSgemmStridedBatched
	sensoryAttentionScoreMatrixCPU = new float[SENSORY_ATTENTION_SCORE_MATRIX_SIZE];
	sensoryAttentionParameterMatrixCPU = new float[SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE];
	cudaMemcpy(sensoryAttentionParameterMatrixCPU, sensoryAttentionParameterMatrixGPU, SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	sensoryKeysMatrixCPU = sensoryAttentionParameterMatrixCPU + QUERY_DIMENSION;
	focusQueriesMatrixCPU = sensoryAttentionParameterMatrixCPU + SENSORY_ATTENTION_PARAMETERS * GLOBAL_SENSORY_VECTORS;
	cpuSgemmStridedBatched(true, false,
		LOCAL_SENSORY_VECTORS, FOCUSES, QUERY_DIMENSION,
		&ONE,
		sensoryKeysMatrixCPU, SENSORY_ATTENTION_PARAMETERS, DYNAMIC_PARAMETERS,
		focusQueriesMatrixCPU, SENSORY_ATTENTION_PARAMETERS, DYNAMIC_PARAMETERS,
		&ZERO,
		sensoryAttentionScoreMatrixCPU, LOCAL_SENSORY_VECTORS, DYNAMIC_PARAMETERS,
		AGENTS);

	// print the sensory attention score
	cout << "CPU Sensory Attention Score Matrix:\n";
		for (int i = 0; i < FOCUSES; i++) {
			for (int j = 0; j < LOCAL_SENSORY_VECTORS; j++)
				std::cout << sensoryAttentionScoreMatrixCPU[i * LOCAL_SENSORY_VECTORS + j] << " ";
			cout << "\n";
		}
		cout << "\n";
	delete[] sensoryAttentionScoreMatrixCPU;

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		VALUE_DIMENSION, FOCUSES, LOCAL_SENSORY_VECTORS,
		&ONE,
		sensoryValuesMatrixGPU, SENSORY_ATTENTION_PARAMETERS, DYNAMIC_PARAMETERS,
		sensoryAttentionScoreMatrixGPU, LOCAL_SENSORY_VECTORS, DYNAMIC_PARAMETERS,
		&ZERO,
		contextMatrixGPU, VALUE_DIMENSION, DYNAMIC_PARAMETERS,
		AGENTS);

	/*// print the context matrix
	float* contextMatrixCPU = new float[CONTEXT_MATRIX_SIZE];
	for (int agent = AGENTS; agent--;) {
		cudaMemcpy(contextMatrixCPU, contextMatrixGPU + agent * DYNAMIC_PARAMETERS, CONTEXT_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Context Matrix:\n";
		for (int i = 0; i < VALUE_DIMENSION; i++) {
			for (int j = 0; j < FOCUSES; j++)
				std::cout << contextMatrixCPU[i * FOCUSES + j] << " ";
			cout << "\n";
		}
		cout << "\n";
	}
	delete[] contextMatrixCPU;*/

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		CONTEXT_ATTENTION_PARAMETERS, FOCUSES, VALUE_DIMENSION,
		&ONE,
		contextAttentionWeightsMatrixGPU, CONTEXT_ATTENTION_PARAMETERS, ZERO,
		contextMatrixGPU, VALUE_DIMENSION, DYNAMIC_PARAMETERS,
		&ZERO,
		contextAttentionParameterMatrixGPU, CONTEXT_ATTENTION_PARAMETERS, DYNAMIC_PARAMETERS,
		AGENTS);

	/*// print the context attention parameters
	float* contextAttentionParameterMatrixCPU = new float[CONTEXT_ATTENTION_PARAMETER_MATRIX_SIZE];
	for (int agent = AGENTS; agent--;) {
		cudaMemcpy(contextAttentionParameterMatrixCPU, contextAttentionParameterMatrixGPU + agent * DYNAMIC_PARAMETERS, CONTEXT_ATTENTION_PARAMETER_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Context Attention Parameter Matrix:\n";
		for (int i = 0; i < FOCUSES; i++) {
			for (int j = 0; j < CONTEXT_ATTENTION_PARAMETERS; j++)
				std::cout << contextAttentionParameterMatrixCPU[i * CONTEXT_ATTENTION_PARAMETERS + j] << " ";
			cout << "\n";
		}
		cout << "\n";
	}
	delete[] contextAttentionParameterMatrixCPU;*/

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		FOCUSES, LOCAL_SENSORY_VECTORS, QUERY_DIMENSION,
		&ONE,
		contextAttentionParameterMatrixGPU, CONTEXT_ATTENTION_PARAMETERS, DYNAMIC_PARAMETERS,
		sensoryAttentionParameterMatrixGPU, SENSORY_ATTENTION_PARAMETERS, DYNAMIC_PARAMETERS,
		&ZERO,
		contextAttentionScoreMatrixGPU, FOCUSES, DYNAMIC_PARAMETERS,
		AGENTS);

	/*// print the context attention scores
	float* contextAttentionScoreMatrixCPU = new float[CONTEXT_ATTENTION_SCORE_MATRIX_SIZE];
	for (int agent = AGENTS; agent--;) {
		cudaMemcpy(contextAttentionScoreMatrixCPU, contextAttentionScoreMatrixGPU + agent * DYNAMIC_PARAMETERS, CONTEXT_ATTENTION_SCORE_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Context Attention Score Matrix:\n";
		for (int i = 0; i < LOCAL_SENSORY_VECTORS; i++) {
			for (int j = 0; j < FOCUSES; j++)
				std::cout << contextAttentionScoreMatrixCPU[i * FOCUSES + j] << " ";
			cout << "\n";
		}
		cout << "\n";
	}
	delete[] contextAttentionScoreMatrixCPU;*/
	
	cudaFree(workspaceGPU);

	return 0;
}