#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <chrono>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

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

	const uint32_t MEMORIES = 7;			// Number of memories per agent
	const uint32_t STIMULI = 3;				// Number of stimuli per agent
	const uint32_t FOCUSES = 5;				// Number of focuses per agent
	
	const uint32_t SENSORY_DIMENSION = 3;	// vector length in each memory, stimulus, and focus
	const uint32_t QUERY_DIMENSION = 7;		// vector length in each query
	const uint32_t VALUE_DIMENSION = 5;		// vector length in each value

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
	const uint32_t SENSORY_ATTENTION_SCORE_MATRIX_SIZE = LOCAL_SENSORY_VECTORS * FOCUSES;							// size of sensory attention result matrix
	const uint32_t CONTEXT_MATRIX_SIZE = VALUE_DIMENSION * FOCUSES;													// size of context matrix
	const uint32_t CONTEXT_ATTENTION_PARAMETER_MATRIX_SIZE = CONTEXT_ATTENTION_PARAMETERS * FOCUSES;				// size of context key and value matrix
	const uint32_t CONTEXT_ATTENTION_SCORE_MATRIX_SIZE = FOCUSES * LOCAL_SENSORY_VECTORS;							// size of context attention result matrix
	const uint32_t SELF_UPDATE_MATRIX = SENSORY_DIMENSION * LOCAL_SENSORY_VECTORS;									// size of self update matrix
	const uint32_t ACTION_REPRESENTAION_SCORE_MATRIX_SIZE = ACTION_VECTORS * GLOBAL_SENSORY_VECTORS;				// size of action matrix
	const uint32_t ACTION_VECTOR_SIZE = ACTIONS;																	// size of action probability output

	const uint32_t staticParameters = INITIAL_SENSORY_MATRIX_SIZE + SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZE + CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZE + ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZE;
	const uint32_t dynamicParameters = SENSORY_MATRIX_SIZE + SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE + SENSORY_ATTENTION_SCORE_MATRIX_SIZE + CONTEXT_MATRIX_SIZE + CONTEXT_ATTENTION_PARAMETER_MATRIX_SIZE + CONTEXT_ATTENTION_SCORE_MATRIX_SIZE + SELF_UPDATE_MATRIX + ACTION_REPRESENTAION_SCORE_MATRIX_SIZE + ACTION_VECTOR_SIZE;

	float* GPUWorkspace;
	cudaMalloc(&GPUWorkspace, staticParameters + dynamicParameters * AGENTS * sizeof(float));

	float* SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZEGPU = GPUWorkspace;
	float* CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZEGPU = SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZEGPU + SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZE + (SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZE & 1);
	float* ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZEGPU = CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZEGPU + CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZE + (CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZE & 1);

	float* SENSORY_MATRIX_SIZEGPU = ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZEGPU + ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZE;
	float* SENSORY_ATTENTION_PARAMETER_MATRIX_SIZEGPU = SENSORY_MATRIX_SIZEGPU + SENSORY_MATRIX_SIZE;
	float* SENSORY_ATTENTION_SCORE_MATRIX_SIZEGPU = SENSORY_ATTENTION_PARAMETER_MATRIX_SIZEGPU + SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE;
	float* CONTEXT_MATRIX_SIZEGPU = SENSORY_ATTENTION_SCORE_MATRIX_SIZEGPU + SENSORY_ATTENTION_SCORE_MATRIX_SIZE;
	float* contextAttentionValuesMatrixGPU = CONTEXT_MATRIX_SIZEGPU + CONTEXT_MATRIX_SIZE;
	float* contextAttentionMatrixGPU = contextAttentionValuesMatrixGPU + CONTEXT_ATTENTION_PARAMETER_MATRIX_SIZE;
	float* SELF_UPDATE_MATRIXGPU = contextAttentionMatrixGPU + CONTEXT_ATTENTION_SCORE_MATRIX_SIZE;
	float* actionValuesMatrixGPU = SELF_UPDATE_MATRIXGPU + SELF_UPDATE_MATRIX;
	float* actionsVectorGPU = actionValuesMatrixGPU + ACTION_REPRESENTAION_SCORE_MATRIX_SIZE;

	float* focusQueryMatrixGPU = SENSORY_MATRIX_SIZEGPU + SENSORY_ATTENTION_PARAMETERS * GLOBAL_SENSORY_VECTORS;
	float* dataKeyMatrixGPU = SENSORY_MATRIX_SIZEGPU + QUERY_DIMENSION;
	float* dataValueMatrixGPU = dataKeyMatrixGPU + QUERY_DIMENSION;

	curandGenerateNormal(curandHandle, SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZEGPU, SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZE + (SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZE & 1), 0.0f, 0.4f);
	curandGenerateNormal(curandHandle, CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZEGPU, CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZE + (CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZE & 1), 0.0f, 0.4f);
	curandGenerateNormal(curandHandle, ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZEGPU, ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZE + (ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZE & 1), 0.0f, 0.4f);

	//print SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZEGPU
	float* SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZECPU = new float[SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZE];
	cudaMemcpy(SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZECPU, SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZEGPU, SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Agent 0 Context Attention Matrix:" << endl;
	for (uint32_t i = 0; i < SENSORY_DIMENSION; i++)
	{
		for (uint32_t j = 0; j < SENSORY_ATTENTION_PARAMETERS; j++)
			cout << SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZECPU[i * SENSORY_ATTENTION_PARAMETERS + j] << " ";
		cout << endl;
	}
	cout << endl;
	delete[] SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZECPU;

	//print CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZEGPU
	float* CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZECPU = new float[CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZE];
	cudaMemcpy(CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZECPU, CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZEGPU, CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Agent 0 Context Attention Matrix:" << endl;
	for (uint32_t i = 0; i < VALUE_DIMENSION; i++)
	{
		for (uint32_t j = 0; j < CONTEXT_ATTENTION_PARAMETERS; j++)
			cout << CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZECPU[i * CONTEXT_ATTENTION_PARAMETERS + j] << " ";
		cout << endl;
	}
	cout << endl;
	delete[] CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZECPU;

	//print ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZEGPU
	float* ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZECPU = new float[ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZE];
	cudaMemcpy(ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZECPU, ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZEGPU, ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Agent 0 Action Matrix:" << endl;
	for (uint32_t i = 0; i < SENSORY_DIMENSION; i++)
	{
		for (uint32_t j = 0; j < ACTION_VECTORS; j++)
			cout << ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZECPU[i * ACTION_VECTORS + j] << " ";
		cout << endl;
	}
	cout << endl;
	delete[] ACTION_REPRESENTAION_WEIGHTS_MATRIX_SIZECPU;

	for (uint32_t agent = AGENTS; agent--;)
		curandGenerateNormal(curandHandle, SENSORY_MATRIX_SIZEGPU + agent * dynamicParameters, SENSORY_MATRIX_SIZE, 0.0f, 0.4f);
	
	//print SENSORY_MATRIX_SIZEGPU
	float* SENSORY_MATRIX_SIZECPU = new float[SENSORY_MATRIX_SIZE];
	for (uint32_t agent = AGENTS; agent--;)
	{
		cudaMemcpy(SENSORY_MATRIX_SIZECPU, SENSORY_MATRIX_SIZEGPU + agent * dynamicParameters, SENSORY_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Data Matrix:" << endl;
		for (uint32_t i = 0; i < SENSORY_DIMENSION; i++)
		{
			for (uint32_t j = 0; j < LOCAL_SENSORY_VECTORS; j++)
				cout << SENSORY_MATRIX_SIZECPU[i * LOCAL_SENSORY_VECTORS + j] << " ";
			cout << endl;
		}
		cout << endl;
	}
	delete[] SENSORY_MATRIX_SIZECPU;

	/*cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		SENSORY_ATTENTION_PARAMETERS, LOCAL_SENSORY_VECTORS, SENSORY_DIMENSION,
		&ONE,
		SENSORY_ATTENTION_WEIGHTS_MATRIX_SIZEGPU, SENSORY_ATTENTION_PARAMETERS, ZERO,
		SENSORY_MATRIX_SIZEGPU, SENSORY_DIMENSION, dynamicParameters,
		&ZERO,
		SENSORY_ATTENTION_PARAMETER_MATRIX_SIZEGPU, SENSORY_ATTENTION_PARAMETERS, dynamicParameters,
		AGENTS);

	//print SENSORY_ATTENTION_PARAMETER_MATRIX_SIZEGPU
	float* SENSORY_ATTENTION_PARAMETER_MATRIX_SIZECPU = new float[SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE];
	for (uint32_t agent = AGENTS; agent--;)
	{
		cudaMemcpy(SENSORY_ATTENTION_PARAMETER_MATRIX_SIZECPU, SENSORY_ATTENTION_PARAMETER_MATRIX_SIZEGPU + agent * dynamicParameters, SENSORY_ATTENTION_PARAMETER_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Context Attention Matrix:" << endl;
		for (uint32_t i = 0; i < LOCAL_SENSORY_VECTORS; i++)
		{
			for (uint32_t j = 0; j < SENSORY_ATTENTION_PARAMETERS; j++)
				cout << SENSORY_ATTENTION_PARAMETER_MATRIX_SIZECPU[i * SENSORY_ATTENTION_PARAMETERS + j] << " ";
			cout << endl;
		}
		cout << endl;
	}
	delete[] SENSORY_ATTENTION_PARAMETER_MATRIX_SIZECPU;

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		LOCAL_SENSORY_VECTORS, FOCUSES, QUERY_DIMENSION,
		&ONE,
		dataKeyMatrixGPU, SENSORY_ATTENTION_PARAMETERS, dynamicParameters,
		focusQueryMatrixGPU, SENSORY_ATTENTION_PARAMETERS, dynamicParameters,
		&ZERO,
		SENSORY_ATTENTION_SCORE_MATRIX_SIZEGPU, LOCAL_SENSORY_VECTORS, dynamicParameters,
		AGENTS);

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		VALUE_DIMENSION, FOCUSES, LOCAL_SENSORY_VECTORS,
		&ONE,
		dataValueMatrixGPU, SENSORY_ATTENTION_PARAMETERS, dynamicParameters,
		SENSORY_ATTENTION_SCORE_MATRIX_SIZEGPU, LOCAL_SENSORY_VECTORS, dynamicParameters,
		&ZERO,
		CONTEXT_MATRIX_SIZEGPU, VALUE_DIMENSION, dynamicParameters,
		AGENTS);

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		CONTEXT_ATTENTION_PARAMETERS, FOCUSES, VALUE_DIMENSION,
		&ONE,
		CONTEXT_ATTENTION_WEIGHTS_MATRIX_SIZEGPU, CONTEXT_ATTENTION_PARAMETERS, ZERO,
		CONTEXT_MATRIX_SIZEGPU, VALUE_DIMENSION, dynamicParameters,
		&ZERO,
		contextAttentionValuesMatrixGPU, CONTEXT_ATTENTION_PARAMETERS, dynamicParameters,
		AGENTS);

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		FOCUSES, LOCAL_SENSORY_VECTORS, QUERY_DIMENSION,
		&ONE,
		contextAttentionValuesMatrixGPU, CONTEXT_ATTENTION_PARAMETERS, dynamicParameters,
		SENSORY_MATRIX_SIZEGPU, SENSORY_ATTENTION_PARAMETERS, dynamicParameters,
		&ZERO,
		contextAttentionMatrixGPU, FOCUSES, dynamicParameters,
		AGENTS);

	//print contextAttentionMatrixGPU
	float* contextAttentionMatrixCPU = new float[CONTEXT_ATTENTION_SCORE_MATRIX_SIZE];
	for (uint32_t agent = AGENTS; agent--;)
	{
		cudaMemcpy(contextAttentionMatrixCPU, contextAttentionMatrixGPU + agent * dynamicParameters, CONTEXT_ATTENTION_SCORE_MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Context Attention Matrix:" << endl;
		for (uint32_t i = 0; i < LOCAL_SENSORY_VECTORS; i++)
		{
			for (uint32_t j = 0; j < FOCUSES; j++)
				cout << contextAttentionMatrixCPU[i * FOCUSES + j] << " ";
			cout << endl;
		}
		cout << endl;
	}
	delete[] contextAttentionMatrixCPU;*/
	
	cudaFree(GPUWorkspace);

	return 0;
}