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

	const uint32_t NUMBER_OF_ACTIONS = 5;
	const uint32_t NUMBER_OF_AGENTS = 3;
	
	const uint32_t DATA_VECTOR_DIMENSION = 3;
	const uint32_t QUERY_VECTOR_DIMENSION = 7;
	const uint32_t VALUE_VECTOR_DIMENSION = 5;

	const uint32_t NUMBER_OF_MEMORIES = 7;
	const uint32_t NUMBER_OF_INPUTS = 5;
	const uint32_t NUMBER_OF_FOCUSES = 3;

	const uint32_t NUMBER_OF_GLOBAL_DATAS = NUMBER_OF_MEMORIES + NUMBER_OF_INPUTS;
	const uint32_t NUMBER_OF_DATAS = NUMBER_OF_GLOBAL_DATAS + NUMBER_OF_FOCUSES;
	const uint32_t NUMBER_OF_ATTENTION_PARAMETERS = 2 * QUERY_VECTOR_DIMENSION + VALUE_VECTOR_DIMENSION;
	const uint32_t NUMBER_OF_CONTEXT_ATTENTION_PARAMETERS = QUERY_VECTOR_DIMENSION + DATA_VECTOR_DIMENSION;
	const uint32_t NUMBER_OF_ACTION_DATAS = NUMBER_OF_ACTIONS + 1;

	const uint32_t attentionWeightsMatrix = NUMBER_OF_ATTENTION_PARAMETERS * DATA_VECTOR_DIMENSION;
	const uint32_t contextAttentionWeightsMatrix = NUMBER_OF_CONTEXT_ATTENTION_PARAMETERS * VALUE_VECTOR_DIMENSION;
	const uint32_t actionWeightsMatrix = NUMBER_OF_ACTION_DATAS * DATA_VECTOR_DIMENSION;

	const uint32_t dataMatrix = DATA_VECTOR_DIMENSION * NUMBER_OF_DATAS;
	const uint32_t attentionValuesMatrix = NUMBER_OF_ATTENTION_PARAMETERS * NUMBER_OF_DATAS;
	const uint32_t attentionMatrix = NUMBER_OF_DATAS * NUMBER_OF_FOCUSES;
	const uint32_t contextMatrix = VALUE_VECTOR_DIMENSION * NUMBER_OF_FOCUSES;
	const uint32_t contextAttentionValuesMatrix = NUMBER_OF_CONTEXT_ATTENTION_PARAMETERS * NUMBER_OF_FOCUSES;
	const uint32_t contextAttentionMatrix = NUMBER_OF_FOCUSES * NUMBER_OF_DATAS;
	const uint32_t updateMatrix = DATA_VECTOR_DIMENSION * NUMBER_OF_DATAS;
	const uint32_t actionValuesMatrix = NUMBER_OF_ACTION_DATAS * NUMBER_OF_GLOBAL_DATAS;
	const uint32_t actionsVector = NUMBER_OF_ACTIONS;

	const uint32_t staticParameters = dataMatrix + attentionWeightsMatrix + contextAttentionWeightsMatrix + actionWeightsMatrix;
	uint32_t dynamicParameters = dataMatrix + attentionValuesMatrix + attentionMatrix + contextMatrix + contextAttentionValuesMatrix + contextAttentionMatrix + updateMatrix + actionValuesMatrix + actionsVector;

	float* GPUWorkspace;
	cudaMalloc(&GPUWorkspace, staticParameters + dynamicParameters * NUMBER_OF_AGENTS * sizeof(float));

	float* attentionWeightsMatrixGPU = GPUWorkspace;
	float* contextAttentionWeightsMatrixGPU = attentionWeightsMatrixGPU + attentionWeightsMatrix + (attentionWeightsMatrix & 1);
	float* actionWeightsMatrixGPU = contextAttentionWeightsMatrixGPU + contextAttentionWeightsMatrix + (contextAttentionWeightsMatrix & 1);

	float* dataMatrixGPU = actionWeightsMatrixGPU + actionWeightsMatrix;
	float* attentionValuesMatrixGPU = dataMatrixGPU + dataMatrix;
	float* attentionMatrixGPU = attentionValuesMatrixGPU + attentionValuesMatrix;
	float* contextMatrixGPU = attentionMatrixGPU + attentionMatrix;
	float* contextAttentionValuesMatrixGPU = contextMatrixGPU + contextMatrix;
	float* contextAttentionMatrixGPU = contextAttentionValuesMatrixGPU + contextAttentionValuesMatrix;
	float* updateMatrixGPU = contextAttentionMatrixGPU + contextAttentionMatrix;
	float* actionValuesMatrixGPU = updateMatrixGPU + updateMatrix;
	float* actionsVectorGPU = actionValuesMatrixGPU + actionValuesMatrix;

	float* focusQueryMatrixGPU = dataMatrixGPU + NUMBER_OF_ATTENTION_PARAMETERS * NUMBER_OF_GLOBAL_DATAS;
	float* dataKeyMatrixGPU = dataMatrixGPU + QUERY_VECTOR_DIMENSION;
	float* dataValueMatrixGPU = dataKeyMatrixGPU + QUERY_VECTOR_DIMENSION;

	curandGenerateNormal(curandHandle, attentionWeightsMatrixGPU, attentionWeightsMatrix + (attentionWeightsMatrix & 1), 0.0f, 0.4f);
	curandGenerateNormal(curandHandle, contextAttentionWeightsMatrixGPU, contextAttentionWeightsMatrix + (contextAttentionWeightsMatrix & 1), 0.0f, 0.4f);
	curandGenerateNormal(curandHandle, actionWeightsMatrixGPU, actionWeightsMatrix + (actionWeightsMatrix & 1), 0.0f, 0.4f);

	//print attentionWeightsMatrixGPU
	float* attentionWeightsMatrixCPU = new float[attentionWeightsMatrix];
	cudaMemcpy(attentionWeightsMatrixCPU, attentionWeightsMatrixGPU, attentionWeightsMatrix * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Agent 0 Context Attention Matrix:" << endl;
	for (uint32_t i = 0; i < DATA_VECTOR_DIMENSION; i++)
	{
		for (uint32_t j = 0; j < NUMBER_OF_ATTENTION_PARAMETERS; j++)
			cout << attentionWeightsMatrixCPU[i * NUMBER_OF_ATTENTION_PARAMETERS + j] << " ";
		cout << endl;
	}
	cout << endl;
	delete[] attentionWeightsMatrixCPU;

	//print contextAttentionWeightsMatrixGPU
	float* contextAttentionWeightsMatrixCPU = new float[contextAttentionWeightsMatrix];
	cudaMemcpy(contextAttentionWeightsMatrixCPU, contextAttentionWeightsMatrixGPU, contextAttentionWeightsMatrix * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Agent 0 Context Attention Matrix:" << endl;
	for (uint32_t i = 0; i < VALUE_VECTOR_DIMENSION; i++)
	{
		for (uint32_t j = 0; j < NUMBER_OF_CONTEXT_ATTENTION_PARAMETERS; j++)
			cout << contextAttentionWeightsMatrixCPU[i * NUMBER_OF_CONTEXT_ATTENTION_PARAMETERS + j] << " ";
		cout << endl;
	}
	cout << endl;
	delete[] contextAttentionWeightsMatrixCPU;

	//print actionWeightsMatrixGPU
	float* actionWeightsMatrixCPU = new float[actionWeightsMatrix];
	cudaMemcpy(actionWeightsMatrixCPU, actionWeightsMatrixGPU, actionWeightsMatrix * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Agent 0 Action Matrix:" << endl;
	for (uint32_t i = 0; i < DATA_VECTOR_DIMENSION; i++)
	{
		for (uint32_t j = 0; j < NUMBER_OF_ACTION_DATAS; j++)
			cout << actionWeightsMatrixCPU[i * NUMBER_OF_ACTION_DATAS + j] << " ";
		cout << endl;
	}
	cout << endl;
	delete[] actionWeightsMatrixCPU;

	for (uint32_t agent = NUMBER_OF_AGENTS; agent--;)
		curandGenerateNormal(curandHandle, dataMatrixGPU + agent * dynamicParameters, dataMatrix, 0.0f, 0.4f);
	
	//print dataMatrixGPU
	float* dataMatrixCPU = new float[dataMatrix];
	for (uint32_t agent = NUMBER_OF_AGENTS; agent--;)
	{
		cudaMemcpy(dataMatrixCPU, dataMatrixGPU + agent * dynamicParameters, dataMatrix * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Data Matrix:" << endl;
		for (uint32_t i = 0; i < DATA_VECTOR_DIMENSION; i++)
		{
			for (uint32_t j = 0; j < NUMBER_OF_DATAS; j++)
				cout << dataMatrixCPU[i * NUMBER_OF_DATAS + j] << " ";
			cout << endl;
		}
		cout << endl;
	}
	delete[] dataMatrixCPU;

	/*cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		NUMBER_OF_ATTENTION_PARAMETERS, NUMBER_OF_DATAS, DATA_VECTOR_DIMENSION,
		&ONE,
		attentionWeightsMatrixGPU, NUMBER_OF_ATTENTION_PARAMETERS, ZERO,
		dataMatrixGPU, DATA_VECTOR_DIMENSION, dynamicParameters,
		&ZERO,
		attentionValuesMatrixGPU, NUMBER_OF_ATTENTION_PARAMETERS, dynamicParameters,
		NUMBER_OF_AGENTS);

	//print attentionValuesMatrixGPU
	float* attentionValuesMatrixCPU = new float[attentionValuesMatrix];
	for (uint32_t agent = NUMBER_OF_AGENTS; agent--;)
	{
		cudaMemcpy(attentionValuesMatrixCPU, attentionValuesMatrixGPU + agent * dynamicParameters, attentionValuesMatrix * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Context Attention Matrix:" << endl;
		for (uint32_t i = 0; i < NUMBER_OF_DATAS; i++)
		{
			for (uint32_t j = 0; j < NUMBER_OF_ATTENTION_PARAMETERS; j++)
				cout << attentionValuesMatrixCPU[i * NUMBER_OF_ATTENTION_PARAMETERS + j] << " ";
			cout << endl;
		}
		cout << endl;
	}
	delete[] attentionValuesMatrixCPU;

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		NUMBER_OF_DATAS, NUMBER_OF_FOCUSES, QUERY_VECTOR_DIMENSION,
		&ONE,
		dataKeyMatrixGPU, NUMBER_OF_ATTENTION_PARAMETERS, dynamicParameters,
		focusQueryMatrixGPU, NUMBER_OF_ATTENTION_PARAMETERS, dynamicParameters,
		&ZERO,
		attentionMatrixGPU, NUMBER_OF_DATAS, dynamicParameters,
		NUMBER_OF_AGENTS);

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		VALUE_VECTOR_DIMENSION, NUMBER_OF_FOCUSES, NUMBER_OF_DATAS,
		&ONE,
		dataValueMatrixGPU, NUMBER_OF_ATTENTION_PARAMETERS, dynamicParameters,
		attentionMatrixGPU, NUMBER_OF_DATAS, dynamicParameters,
		&ZERO,
		contextMatrixGPU, VALUE_VECTOR_DIMENSION, dynamicParameters,
		NUMBER_OF_AGENTS);

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		NUMBER_OF_CONTEXT_ATTENTION_PARAMETERS, NUMBER_OF_FOCUSES, VALUE_VECTOR_DIMENSION,
		&ONE,
		contextAttentionWeightsMatrixGPU, NUMBER_OF_CONTEXT_ATTENTION_PARAMETERS, ZERO,
		contextMatrixGPU, VALUE_VECTOR_DIMENSION, dynamicParameters,
		&ZERO,
		contextAttentionValuesMatrixGPU, NUMBER_OF_CONTEXT_ATTENTION_PARAMETERS, dynamicParameters,
		NUMBER_OF_AGENTS);

	cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
		NUMBER_OF_FOCUSES, NUMBER_OF_DATAS, QUERY_VECTOR_DIMENSION,
		&ONE,
		contextAttentionValuesMatrixGPU, NUMBER_OF_CONTEXT_ATTENTION_PARAMETERS, dynamicParameters,
		dataMatrixGPU, NUMBER_OF_ATTENTION_PARAMETERS, dynamicParameters,
		&ZERO,
		contextAttentionMatrixGPU, NUMBER_OF_FOCUSES, dynamicParameters,
		NUMBER_OF_AGENTS);

	//print contextAttentionMatrixGPU
	float* contextAttentionMatrixCPU = new float[contextAttentionMatrix];
	for (uint32_t agent = NUMBER_OF_AGENTS; agent--;)
	{
		cudaMemcpy(contextAttentionMatrixCPU, contextAttentionMatrixGPU + agent * dynamicParameters, contextAttentionMatrix * sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Agent " << agent << " Context Attention Matrix:" << endl;
		for (uint32_t i = 0; i < NUMBER_OF_DATAS; i++)
		{
			for (uint32_t j = 0; j < NUMBER_OF_FOCUSES; j++)
				cout << contextAttentionMatrixCPU[i * NUMBER_OF_FOCUSES + j] << " ";
			cout << endl;
		}
		cout << endl;
	}
	delete[] contextAttentionMatrixCPU;*/
	
	cudaFree(GPUWorkspace);

	return 0;
}