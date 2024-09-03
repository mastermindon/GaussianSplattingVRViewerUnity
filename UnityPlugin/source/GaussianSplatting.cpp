#include "GaussianSplatting.h"
#include "GaussianSplatting.h"
#include "CudaKernels.h"
#include <cuda_runtime.h>
#include <rasterizer.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace std;

typedef	Eigen::Matrix<int, 3, 1, Eigen::DontAlign> Vector3i;

inline float sigmoid(const float m1) { return 1.0f / (1.0f + exp(-m1)); }

static constexpr float NormalFloatData(const float& v)
{
	//return std::clamp(v, 0.0f, 1.0f);
	return v;
}

inline std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

template<typename T> float* append_cuda(float* cuda, size_t sz, vector<T>& data) {
	float* ncuda = nullptr;
	size_t snb = sizeof(T) * data.size();
	size_t size = sizeof(T) * sz;

	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&ncuda, size + snb));
	if (cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(ncuda, cuda, size, cudaMemcpyDeviceToDevice)); }
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(((char*)ncuda) + size, data.data(), snb, cudaMemcpyHostToDevice));
	if (cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree(cuda)); }
	return ncuda;
}

template<typename T> float* remove_cuda(float* cuda, size_t sz, size_t pos, size_t nb) {
	if (cuda == nullptr) { return nullptr; }

	float* ncuda = nullptr;
	size_t snb = sizeof(T) * nb;
	size_t spos = sizeof(T) * pos;
	size_t size = sizeof(T) * sz;
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&ncuda, size - snb));
	if (spos > 0) {
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(ncuda, cuda, spos, cudaMemcpyDeviceToDevice));
	}
	if (spos + snb < size) {
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(((char*)ncuda + spos), ((char*)cuda) + spos + snb, size - snb - spos, cudaMemcpyDeviceToDevice));
	}
	CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)cuda));
	return ncuda;
}

//Gaussian Splatting data structure
template<int D>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
};

#pragma pack(push) // 保存当前对齐状态
#pragma pack(1)    // 设置一字节对齐
template<int D>
struct HighPoint
{
	Vector3f pos;
	//uint n;
	uint32_t shs[D];
	uint64_t scale;
	//Vector4f rot;
	//uint32_t rot;
	uint64_t rot;
	uint32_t color;
};
#pragma pack(pop) // 恢复之前的对齐状态

#pragma pack(push) // 保存当前对齐状态
#pragma pack(1)    // 设置一字节对齐
template<int D>
struct MiddlePoint
{
	//uint32_t pos;
	uint64_t pos;
	//uint n;
	uint16_t shs[D];
	//float opacity;
	uint32_t scale;
	uint32_t rot;
	uint32_t color;
	//uint64_t rot;
};
#pragma pack(pop) // 恢复之前的对齐状态

struct MiddleCompressData
{
	Vector3f chunkMinpos;
	Vector3f chunkMinshs;
	Vector3f chunkMinscl;
	Vector4f chunkMinrot;
	Vector4f chunkMincol;

	Vector3f chunkMaxpos;
	Vector3f chunkMaxshs;
	Vector3f chunkMaxscl;
	Vector4f chunkMaxrot;
	Vector4f chunkMaxcol;
};

template<int D> int loadPly(const char* filename, const char* fileMd5,std::vector<Pos>& pos, std::vector<SHs<3>>& shs, std::vector<float>& opacities, std::vector<Scale>& scales, std::vector<Rot>& rot, Vector3f& minn, Vector3f& maxx) throw(std::bad_exception);
template<int D> int loadPlyCompress(const char* filename, const char* fileMd5, std::vector<Pos>& pos, std::vector<SHs<3>>& shs, std::vector<float>& opacities, std::vector<Scale>& scales, std::vector<Rot>& rot, Vector3f& minn, Vector3f& maxx) throw(std::bad_exception);

void GaussianSplattingRenderer::SetModelCrop(int model, float* box_min, float* box_max) {
	for (std::list<SplatModel>::iterator it = models.begin(); it != models.end(); ++it) {
		if (it->index == model) {
			it->_boxmin = Vector3f(box_min);
			it->_boxmax = Vector3f(box_max);
			break;
		}
	}
}

void GaussianSplattingRenderer::GetModelCrop(int model, float* box_min, float* box_max) {
	for (std::list<SplatModel>::iterator it = models.begin(); it != models.end(); ++it) {
		if (it->index == model) {
			box_min[0] = it->_scenemin.x();
			box_min[1] = it->_scenemin.y();
			box_min[2] = it->_scenemin.z();
			box_max[0] = it->_scenemax.x();
			box_max[1] = it->_scenemax.y();
			box_max[2] = it->_scenemax.z();
			break;
		}
	}
}

int GaussianSplattingRenderer::GetNbSplat() {
	return count;
}

void GaussianSplattingRenderer::SetProcessInfo(const char* fileMd5,float process)
{
	const std::lock_guard<std::mutex> lock(cuda_mtx);

	std::string md5 = fileMd5;
	FileConfig* config = nullptr;

	if (fileConfigData.find(fileMd5) != fileConfigData.end())
	{
		config = fileConfigData[fileMd5];
	}

	if (modelLoadProcess != nullptr)
	{
		modelLoadProcess(config->fileName, config->fileMd5, process);
	}
}

void GaussianSplattingRenderer::SetLoadedFileConfig(const char* fileName, const char* fileMd5, int loadedType)
{
	const std::lock_guard<std::mutex> lock(cuda_mtx);

	std::string md5 = fileMd5;
	FileConfig* config = new FileConfig;
	config->fileName = fileName;
	config->fileMd5 = fileMd5;
	config->loadedType = loadedType;
	fileConfigData[md5] = config;
}

void GaussianSplattingRenderer::Load(const char* file, const char* fileMd5) {
	count_cpu = 0;
	
	// Load the PLY data (AoS) to the GPU (SoA)
	if (_sh_degree == 0)
	{
		count_cpu = loadPly<0>(file, fileMd5, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (_sh_degree == 1)
	{
		count_cpu = loadPly<1>(file, fileMd5, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (_sh_degree == 2)
	{
		count_cpu = loadPly<2>(file, fileMd5, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
	else if (_sh_degree == 3)
	{
		count_cpu = loadPly<3>(file, fileMd5, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
	}
}

void GaussianSplattingRenderer::LoadCompressed(const char* file, const char* fileMd5) {
	count_cpu = 0;
	count_cpu = loadPlyCompress<3>(file, fileMd5, pos, shs, opacity, scale, rot, _scenemin, _scenemax);
}

int GaussianSplattingRenderer::CopyToCuda() {
	if (count_cpu == 0) {
		return 0;
	}

	const std::lock_guard<std::mutex> lock(cuda_mtx);

	//Register new model
	model_idx += 1;
	models.push_back({ model_idx, count_cpu, false, _scenemin, _scenemax, _scenemin, _scenemax });

	pos_cuda = append_cuda(pos_cuda, count, pos);
	rot_cuda = append_cuda(rot_cuda, count, rot);
	shs_cuda = append_cuda(shs_cuda, count, shs);
	opacity_cuda = append_cuda(opacity_cuda, count, opacity);
	scale_cuda = append_cuda(scale_cuda, count, scale);

	//set new size with the appened model
	count += count_cpu;

	//Working buffer or fixed data
	//can be fully reallocated
	if (background_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)background_cuda)); }
	if (rect_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)rect_cuda)); }
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * count * sizeof(int)));

	bool white_bg = false;
	float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));
	
	AllocateRenderContexts();

	//Update count and return new model index
	return model_idx;
}

void GaussianSplattingRenderer::RemoveModel(int model) {
	const std::lock_guard<std::mutex> lock(cuda_mtx);
	
	size_t start = 0;
	std::list<SplatModel>::iterator mit = models.end();
	for (std::list<SplatModel>::iterator it = models.begin(); it != models.end(); ++it) {
		if (it->index == model) {
			mit = it;
			break;
		}
		start += it->size;
	}

	if (mit != models.end()) {
		size_t size = mit->size;
		pos_cuda = remove_cuda<Pos>(pos_cuda, count, start, size);
		rot_cuda = remove_cuda<Rot>(rot_cuda, count, start, size);
		shs_cuda = remove_cuda<SHs<3>>(shs_cuda, count, start, size);
		opacity_cuda = remove_cuda<float>(opacity_cuda, count, start, size);
		scale_cuda = remove_cuda<Scale>(scale_cuda, count, start, size);

		count -= size;
		models.erase(mit);

		//Working buffer or fixed data
		//can be fully reallocated
		if (background_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)background_cuda)); }
		if (rect_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)rect_cuda)); }
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * count * sizeof(int)));

		bool white_bg = false;
		float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
		CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

		AllocateRenderContexts();

	} else {
		throw std::runtime_error("Model index not found.");
	}
}

void GaussianSplattingRenderer::CreateRenderContext(int idx) {

	const std::lock_guard<std::mutex> lock(cuda_mtx);

	//Resize the buffers
	geom[idx] = new AllocFuncBuffer;
	binning[idx] = new AllocFuncBuffer;
	img[idx] = new AllocFuncBuffer;
	renData[idx] = new RenderData;

	//Alloc
	geom[idx]->bufferFunc = resizeFunctional(&geom[idx]->ptr, geom[idx]->allocd);
	binning[idx]->bufferFunc = resizeFunctional(&binning[idx]->ptr, binning[idx]->allocd);
	img[idx]->bufferFunc = resizeFunctional(&img[idx]->ptr, img[idx]->allocd);

	//Alloc cuda ressource for view model
	AllocateRenderContexts();
}

void GaussianSplattingRenderer::RemoveRenderContext(int idx) {
	const std::lock_guard<std::mutex> lock(cuda_mtx);
	
	//freee cuda resources
	if (geom.at(idx)->ptr != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)geom.at(idx)->ptr)); }
	if (binning.at(idx)->ptr != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)binning.at(idx)->ptr)); }
	if (img.at(idx)->ptr != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)img.at(idx)->ptr)); }

	geom.erase(idx);
	binning.erase(idx);
	img.erase(idx);

	if (renData.at(idx)->view_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->view_cuda)); }
	if (renData.at(idx)->proj_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->proj_cuda)); }
	if (renData.at(idx)->model_sz != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->model_sz)); }
	if (renData.at(idx)->model_active != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->model_active)); }
	if (renData.at(idx)->cam_pos_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->cam_pos_cuda)); }
	if (renData.at(idx)->boxmin != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->boxmin)); }
	if (renData.at(idx)->boxmax != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->boxmax)); }
	if (renData.at(idx)->frustums != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)renData.at(idx)->frustums)); }

	RenderData* data = renData.at(idx);
	renData.erase(idx);
	delete data;
}

void GaussianSplattingRenderer::AllocateRenderContexts() {
	size_t nb_models = models.size();
	for (auto kv: renData) {
		RenderData* data = kv.second;
		//reallocate only if needed
		if (data->nb_model_allocated != nb_models) {
			data->nb_model_allocated = nb_models;
			
			//free last allocated ressources
			if (data->view_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->view_cuda))); }
			if (data->proj_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->proj_cuda))); }
			if (data->model_sz != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->model_sz))); }
			if (data->model_active != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->model_active))); }
			if (data->cam_pos_cuda != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->cam_pos_cuda))); }
			if (data->boxmin != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->boxmin))); }
			if (data->boxmax != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->boxmax))); }
			if (data->frustums != nullptr) { CUDA_SAFE_CALL_ALWAYS(cudaFree((void*)(data->frustums))); }

			// Create space for view parameters for each model
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->view_cuda), sizeof(Matrix4f) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->proj_cuda), sizeof(Matrix4f) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->model_sz), sizeof(int) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->model_active), sizeof(int) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->cam_pos_cuda), 3 * sizeof(float) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->boxmin), 3 * sizeof(float) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->boxmax), 3 * sizeof(float) * nb_models));
			CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&(data->frustums), 6 * sizeof(float)));
		}
	}
}

void GaussianSplattingRenderer::SetActiveModel(int model, bool active) {
	for (SplatModel& m : models) {
		if (m.index == model) {
			m.active = active;
		}
	}
}

void GaussianSplattingRenderer::Preprocess(int context, const std::map<int, Matrix4f>& view_mat, const std::map<int, Matrix4f>& proj_mat, const std::map<int, Vector3f>& position, Vector6f frumstums, float fovy, int width, int height) {
	//view_mat.row(1) *= -1;
	//view_mat.row(2) *= -1;
	//proj_mat.row(1) *= -1;

	const std::lock_guard<std::mutex> lock(cuda_mtx);

	if (count == 0) { return; }
	
	float aspect_ratio = (float)width / (float)height;
	float tan_fovy = tan(fovy * 0.5f);
	float tan_fovx = tan_fovy * aspect_ratio;

	RenderData* rdata = renData.at(context);
	int nb_models = models.size();
	int midx = 0;
	for (const SplatModel& m : models) {
		int active = (m.active && view_mat.find(m.index) != view_mat.end()) ? 1 : 0;
		int msize = m.size;
		CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->model_sz) + midx * sizeof(int), &msize, sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->model_active) + midx * sizeof(int), &active, sizeof(int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->boxmin) + midx * sizeof(float) * 3, m._boxmin.data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->boxmax) + midx * sizeof(float) * 3, m._boxmax.data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
		if (active == 1) {
			CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->view_cuda) + midx * sizeof(Matrix4f), view_mat.at(m.index).data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->proj_cuda) + midx * sizeof(Matrix4f), proj_mat.at(m.index).data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->cam_pos_cuda) + midx * sizeof(float) * 3, position.at(m.index).data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
		}
		midx += 1;
	}
	CUDA_SAFE_CALL(cudaMemcpy((char*)(rdata->frustums), frumstums.data(), sizeof(float) * 6, cudaMemcpyHostToDevice));

	// Rasterize
	int* rects = _fastCulling ? rect_cuda : nullptr;
	rdata->num_rendered = CudaRasterizer::Rasterizer::forward_preprocess(
		geom.at(context)->bufferFunc,
		binning.at(context)->bufferFunc,
		img.at(context)->bufferFunc,
		count, /*_sh_degree*/_show_sh_degree, 16,
		background_cuda,
		width, height,
		pos_cuda,
		shs_cuda,
		nullptr,
		opacity_cuda,
		scale_cuda,
		_scalingModifier,
		rot_cuda,
		nullptr,
		rdata->view_cuda,
		rdata->proj_cuda,
		rdata->cam_pos_cuda,
		rdata->frustums,
		rdata->model_sz,
		rdata->model_active,
		nb_models,
		tan_fovx,
		tan_fovy,
		false,
		nullptr,
		rects,
		rdata->boxmin,
		rdata->boxmax);
}

void GaussianSplattingRenderer::Render(int context, float* image_cuda, float* depth_cuda, cudaSurfaceObject_t camera_depth_cuda, float fovy, int width, int height) {
	if (count > 0 && renData.at(context)->num_rendered > 0) {
		
		RenderData* rdata = renData.at(context);
		
		const std::lock_guard<std::mutex> lock(cuda_mtx);
		
		float aspect_ratio = (float)width / (float)height;
		float tan_fovy = tan(fovy * 0.5f);
		float tan_fovx = tan_fovy * aspect_ratio;

		int* rects = _fastCulling ? rect_cuda : nullptr;

		CudaRasterizer::Rasterizer::forward_render(
			geom.at(context)->bufferFunc,
			binning.at(context)->bufferFunc,
			img.at(context)->bufferFunc,
			count, _sh_degree, 16,
			background_cuda,
			camera_depth_cuda,
			width, height,
			pos_cuda,
			shs_cuda,
			nullptr,
			opacity_cuda,
			scale_cuda,
			_scalingModifier,
			rot_cuda,
			nullptr,
			rdata->view_cuda,
			rdata->proj_cuda,
			rdata->cam_pos_cuda,
			tan_fovx,
			tan_fovy,
			false,
			image_cuda,
			depth_cuda,
			nullptr,
			rects,
			rdata->boxmin,
			rdata->boxmax,
			rdata->num_rendered);
	} else {
		CUDA_SAFE_CALL(cudaMemset(image_cuda, 0, sizeof(float) * 4 * width * height));
		CUDA_SAFE_CALL(cudaMemset(depth_cuda, 0, sizeof(float) * width * height));
	}
}

// Load the Gaussians from the given file.
template<int D>
int loadPly(const char* filename,
	const char* fileMd5,
	std::vector<Pos>& pos,
	std::vector<SHs<3>>& shs,
	std::vector<float>& opacities,
	std::vector<Scale>& scales,
	std::vector<Rot>& rot,
	Vector3f& minn,
	Vector3f& maxx)
{
	float process = 0;
	SetModelProcess(fileMd5, 1);
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		throw std::runtime_error((stringstream() << "Unable to find model's PLY file, attempted:\n" << filename).str());

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int lcount;
	ss >> dummy >> dummy >> lcount;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<RichPoint<D>> points(lcount);
	infile.read((char*)points.data(), lcount * sizeof(RichPoint<D>));
	SetModelProcess(fileMd5,3);
	// Resize our SoA data
	pos.resize(lcount);
	shs.resize(lcount);
	scales.resize(lcount);
	rot.resize(lcount);
	opacities.resize(lcount);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile 
	// (close in 3D --> close in 2D).
	minn = Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < lcount; i++)
	{
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(lcount);
	for (int i = 0; i < lcount; i++)
	{
		Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
		Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		Vector3i xyz = scaled.cast<int>();

		uint64_t code = 0;
		for (int i = 0; i < 21; i++) {
			code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
		}

		mapp[i].first = code;
		mapp[i].second = i;

		SetModelProcess(fileMd5, 3 + i * 1.0f / lcount * 20.0f);
	}
	SetModelProcess(fileMd5, 23);
	auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
		return a.first < b.first;
	};
	std::sort(mapp.begin(), mapp.end(), sorter);
	SetModelProcess(fileMd5, 25);
	// Move data from AoS to SoA
	int SH_N = (D + 1) * (D + 1);
	for (int k = 0; k < lcount; k++)
	{
		int i = mapp[k].second;
		pos[k] = points[i].pos;

		// Normalize quaternion
		float length2 = 0;
		for (int j = 0; j < 4; j++)
			length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
		float length = sqrt(length2);
		for (int j = 0; j < 4; j++)
			rot[k].rot[j] = points[i].rot.rot[j] / length;

		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			scales[k].scale[j] = exp(points[i].scale.scale[j]);

		// Activate alpha
		opacities[k] = sigmoid(points[i].opacity);

		shs[k].shs[0] = points[i].shs.shs[0];
		shs[k].shs[1] = points[i].shs.shs[1];
		shs[k].shs[2] = points[i].shs.shs[2];
		for (int j = 1; j < SH_N; j++)
		{
			shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
			shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
			shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
		}
		SetModelProcess(fileMd5, 25 + k * 1.0f / lcount * 75.0f);
	}
	return lcount;
}
static vector<float> CacheData3f(3);
static vector<float> CacheData4f(4);

vector<float>& DecodePacked_16_16_16_16(uint64_t enc)
{
	CacheData4f[0] = (enc & 65535) / 65535.0;
	CacheData4f[1] = ((enc >> 16) & 65535) / 65535.0;
	CacheData4f[2] = ((enc >> 32) & 65535) / 65535.0;
	CacheData4f[3] = ((enc >> 48) & 65535) / 65535.0;
	return CacheData4f;
}

vector<float>& DecodePacked_8_8_8_8(uint32_t enc)
{
	CacheData4f[0] = (enc & 255) / 255.0;
	CacheData4f[1] = ((enc >> 8) & 255) / 255.0;
	CacheData4f[2] = ((enc >> 16) & 255) / 255.0;
	CacheData4f[3] = ((enc >> 24) & 255) / 255.0;
	return CacheData4f;
}

vector<float>& DecodePacked_21_21_21(uint64_t enc)
{
	CacheData3f[0] = (enc & 2097151) / 2097151.0;
	CacheData3f[1] = ((enc >> 21) & 2097151) / 2097151.0;
	CacheData3f[2] = ((enc >> 42) & 2097151) / 2097151.0;
	return CacheData3f;
}

vector<float>& DecodePacked_11_10_11(uint32_t enc)
{
	CacheData3f[0] = (enc & 2047) / 2047.0;
	CacheData3f[1] = ((enc >> 11) & 1023) / 1023.0;
	CacheData3f[2] = ((enc >> 21) & 2047) / 2047.0;
	return CacheData3f;
}

vector<float>& DecodePacked_5_6_5(uint16_t enc)
{
	CacheData3f[0] = (enc & 31) / 31.0;
	CacheData3f[1] = ((enc >> 5) & 63) / 63.0;
	CacheData3f[2] = ((enc >> 11) & 31) / 31.0;
	return CacheData3f;
}
vector<float>& DecodePacked_10_10_10_2(uint32_t enc)
{
	CacheData4f[0] = (enc & 1023) / 1023.0;
	CacheData4f[1] = ((enc >> 10) & 1023) / 1023.0;
	CacheData4f[2] = ((enc >> 20) & 1023) / 1023.0;
	CacheData4f[3] = ((enc >> 30) & 3) / 3.0;
	return CacheData4f;
}


template <class _Ty>
_NODISCARD constexpr _Ty _Linear_for_lerp(const _Ty _ArgA, const _Ty _ArgB, const _Ty _ArgT) {
	if (_Is_constant_evaluated()) {
		auto _Smaller = _ArgT;
		auto _Larger = _ArgB - _ArgA;
		auto _Abs_smaller = _Float_abs(_Smaller);
		auto _Abs_larger = _Float_abs(_Larger);
		if (_Abs_larger < _Abs_smaller) {
			_STD swap(_Smaller, _Larger);
			_STD swap(_Abs_smaller, _Abs_larger);
		}

		if (_Abs_smaller > 1) {
			// _Larger is too large to be subnormal, so scaling by 0.5 is exact, and the product _Smaller * _Larger is
			// large enough that if _ArgA is subnormal, it will be too small to contribute anyway and this way can
			// sometimes avoid overflow problems.
			return 2 * (_Ty{ 0.5 } *_ArgA + _Smaller * (_Ty{ 0.5 } *_Larger));
		}
		else {
			return _ArgA + _Smaller * _Larger;
		}
	}

	return _STD fma(_ArgT, _ArgB - _ArgA, _ArgA);
}

template <class _Ty>
_NODISCARD constexpr _Ty _Common_lerp(const _Ty _ArgA, const _Ty _ArgB, const _Ty _ArgT) noexcept {
	// on a line intersecting {(0.0, _ArgA), (1.0, _ArgB)}, return the Y value for X == _ArgT

	const bool _T_is_finite = _Is_finite(_ArgT);
	if (_T_is_finite && _Is_finite(_ArgA) && _Is_finite(_ArgB)) {
		// 99% case, put it first; this block comes from P0811R3
		if ((_ArgA <= 0 && _ArgB >= 0) || (_ArgA >= 0 && _ArgB <= 0)) {
			// exact, monotonic, bounded, determinate, and (for _ArgA == _ArgB == 0) consistent:
			return _ArgT * _ArgB + (1 - _ArgT) * _ArgA;
		}

		if (_ArgT == 1) {
			// exact
			return _ArgB;
		}

		// exact at _ArgT == 0, monotonic except near _ArgT == 1, bounded, determinate, and consistent:
		const auto _Candidate = _Linear_for_lerp(_ArgA, _ArgB, _ArgT);
		// monotonic near _ArgT == 1:
		if ((_ArgT > 1) == (_ArgB > _ArgA)) {
			if (_ArgB > _Candidate) {
				return _ArgB;
			}
		}
		else {
			if (_Candidate > _ArgB) {
				return _ArgB;
			}
		}

		return _Candidate;
	}

	if (_Is_constant_evaluated()) {
		if (_Is_nan(_ArgA)) {
			return _ArgA;
		}

		if (_Is_nan(_ArgB)) {
			return _ArgB;
		}

		if (_Is_nan(_ArgT)) {
			return _ArgT;
		}
	}
	else {
		// raise FE_INVALID if at least one of _ArgA, _ArgB, and _ArgT is signaling NaN
		if (_Is_nan(_ArgA) || _Is_nan(_ArgB)) {
			return (_ArgA + _ArgB) + _ArgT;
		}

		if (_Is_nan(_ArgT)) {
			return _ArgT + _ArgT;
		}
	}

	if (_T_is_finite) {
		// _ArgT is finite, _ArgA and/or _ArgB is infinity
		if (_ArgT < 0) {
			// if _ArgT < 0:     return infinity in the "direction" of _ArgA if that exists, NaN otherwise
			return _ArgA - _ArgB;
		}
		else if (_ArgT <= 1) {
			// if _ArgT == 0:    return _ArgA (infinity) if _ArgB is finite, NaN otherwise
			// if 0 < _ArgT < 1: return infinity "between" _ArgA and _ArgB if that exists, NaN otherwise
			// if _ArgT == 1:    return _ArgB (infinity) if _ArgA is finite, NaN otherwise
			return _ArgT * _ArgB + (1 - _ArgT) * _ArgA;
		}
		else {
			// if _ArgT > 1:     return infinity in the "direction" of _ArgB if that exists, NaN otherwise
			return _ArgB - _ArgA;
		}
	}
	else {
		// _ArgT is an infinity; return infinity in the "direction" of _ArgA and _ArgB if that exists, NaN otherwise
		return _ArgT * (_ArgB - _ArgA);
	}
}

_EXPORT_STD _NODISCARD constexpr inline float lerp(const float _ArgA, const float _ArgB, const float _ArgT) noexcept {
	return _Common_lerp(_ArgA, _ArgB, _ArgT);
}

_EXPORT_STD _NODISCARD constexpr inline double lerp(
	const double _ArgA, const double _ArgB, const double _ArgT) noexcept {
	return _Common_lerp(_ArgA, _ArgB, _ArgT);
}

_EXPORT_STD _NODISCARD constexpr inline long double lerp(
	const long double _ArgA, const long double _ArgB, const long double _ArgT) noexcept {
	return _Common_lerp(_ArgA, _ArgB, _ArgT);
}

_EXPORT_STD template <class _Ty1, class _Ty2, class _Ty3,
	enable_if_t<is_arithmetic_v<_Ty1>&& is_arithmetic_v<_Ty2>&& is_arithmetic_v<_Ty3>, int> = 0>
	_NODISCARD constexpr auto lerp(const _Ty1 _ArgA, const _Ty2 _ArgB, const _Ty3 _ArgT) noexcept {
	using _Tgt = conditional_t<_Is_any_of_v<long double, _Ty1, _Ty2, _Ty3>, long double, double>;
	return _Common_lerp(static_cast<_Tgt>(_ArgA), static_cast<_Tgt>(_ArgB), static_cast<_Tgt>(_ArgT));
}
// Load the Gaussians from the given file.
template<int D>
int loadPlyCompress(const char* filename,
	const char* fileMd5,
	std::vector<Pos>& pos,
	std::vector<SHs<3>>& shs,
	std::vector<float>& opacities,
	std::vector<Scale>& scales,
	std::vector<Rot>& rot,
	Vector3f& minn,
	Vector3f& maxx)
{
	SetModelProcess(fileMd5, 1);
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		throw std::runtime_error((stringstream() << "Unable to find model's PLY file, attempted:\n" << filename).str());
	uint8_t quality = 2;//1是高，2是中
	uint64_t lcount =0, configNum = 0,dataNum = 0;
	infile.read((char*)(&quality), 1);
	infile.read((char*)(&lcount) , 8);
	infile.read((char*)(&configNum) , 8);
	infile.read((char*)(&dataNum) , 8);
	std::vector<MiddleCompressData> middleCompressData(configNum);
	infile.read((char*)middleCompressData.data(), configNum * sizeof(MiddleCompressData));
	// Read all Gaussians at once (AoS)
	if (quality == 2)
	{
		std::vector<MiddlePoint<15>> points(lcount);
		infile.read((char*)points.data(), lcount * sizeof(MiddlePoint<15>));
		SetModelProcess(fileMd5, 3);
		// Resize our SoA data
		pos.resize(lcount);
		shs.resize(lcount);
		scales.resize(lcount);
		rot.resize(lcount);
		opacities.resize(lcount);
		//
		int chunkCount = (points.size() + kChunkSize - 1) / kChunkSize;
		//不分块了，按总的来算，多线程就弄多个最大最小值，再一起比较，最终每个变量只取一组最大最小值-----(不分块的话，场景大了，容易失真)
		for (int chunkIdx = 0; chunkIdx < chunkCount; chunkIdx++)
		{
			int splatBegin = MIN(chunkIdx * kChunkSize, points.size());
			int splatEnd = MIN((chunkIdx + 1) * kChunkSize, points.size());
			// calculate data bounds inside the chunk
			MiddleCompressData configData = middleCompressData[chunkIdx];
			// Move data from AoS to SoA
			int SH_N = (D + 1) * (D + 1);
			for (int i = splatBegin; i < splatEnd; i++)
			{
				vector<float>& posData = DecodePacked_21_21_21(points[i].pos);
				pos[i][0] = lerp(configData.chunkMinpos[0], configData.chunkMaxpos[0], posData[0]);
				pos[i][1] = lerp(configData.chunkMinpos[1], configData.chunkMaxpos[1], posData[1]);
				pos[i][2] = lerp(configData.chunkMinpos[2], configData.chunkMaxpos[2], posData[2]);
				//pos[i][0] = lerp(configData.chunkMinpos[0], configData.chunkMaxpos[0], points[i].pos[0]);
				//pos[i][1] = lerp(configData.chunkMinpos[1], configData.chunkMaxpos[1], points[i].pos[1]);
				//pos[i][2] = lerp(configData.chunkMinpos[2], configData.chunkMaxpos[2], points[i].pos[2]);

				vector<float>& scaleData = DecodePacked_11_10_11(points[i].scale);
				scales[i].scale[0] = lerp(configData.chunkMinscl[0], configData.chunkMaxscl[0], scaleData[0]);
				scales[i].scale[1] = lerp(configData.chunkMinscl[1], configData.chunkMaxscl[1], scaleData[1]);
				scales[i].scale[2] = lerp(configData.chunkMinscl[2], configData.chunkMaxscl[2], scaleData[2]);

				// int32
				vector<float>& rotData = DecodePacked_8_8_8_8(points[i].rot);
				rot[i].rot[0] = lerp(configData.chunkMinrot[0], configData.chunkMaxrot[0], rotData[0]);
				rot[i].rot[1] = lerp(configData.chunkMinrot[1], configData.chunkMaxrot[1], rotData[1]);
				rot[i].rot[2] = lerp(configData.chunkMinrot[2], configData.chunkMaxrot[2], rotData[2]);
				rot[i].rot[3] = lerp(configData.chunkMinrot[3], configData.chunkMaxrot[3], rotData[3]);
				/* int64
				vector<float> rotData = DecodePacked_16_16_16_16(points[i].rot);
				rot[i].rot[0] = lerp(configData.chunkMinrot[0], configData.chunkMaxrot[0], rotData[0]);
				rot[i].rot[1] = lerp(configData.chunkMinrot[1], configData.chunkMaxrot[1], rotData[1]);
				rot[i].rot[2] = lerp(configData.chunkMinrot[2], configData.chunkMaxrot[2], rotData[2]);
				rot[i].rot[3] = lerp(configData.chunkMinrot[3], configData.chunkMaxrot[3], rotData[3]);
				*/
				/* float4
				rot[i].rot[0] = lerp(configData.chunkMinrot[0], configData.chunkMaxrot[0], points[i].rot[0]);
				rot[i].rot[1] = lerp(configData.chunkMinrot[1], configData.chunkMaxrot[1], points[i].rot[1]);
				rot[i].rot[2] = lerp(configData.chunkMinrot[2], configData.chunkMaxrot[2], points[i].rot[2]);
				rot[i].rot[3] = lerp(configData.chunkMinrot[3], configData.chunkMaxrot[3], points[i].rot[3]);
				*/
				//color
							// int32
				vector<float>& colorData = DecodePacked_8_8_8_8(points[i].color);
				shs[i].shs[0] = lerp(configData.chunkMincol[0], configData.chunkMaxcol[0], colorData[0]);
				shs[i].shs[1] = lerp(configData.chunkMincol[1], configData.chunkMaxcol[1], colorData[1]);
				shs[i].shs[2] = lerp(configData.chunkMincol[2], configData.chunkMaxcol[2], colorData[2]);
				opacities[i] = lerp(configData.chunkMincol[3], configData.chunkMaxcol[3], colorData[3]);

				for (int j = 1; j < SH_N; j++)
				{
					//vector<float> shsData = DecodePacked_11_10_11(points[i].shs[j - 1]);
					vector<float>& shsData = DecodePacked_5_6_5(points[i].shs[j - 1]);

					shs[i].shs[j * 3 + 0] = lerp(configData.chunkMinshs[0], configData.chunkMaxshs[0], shsData[0]);
					shs[i].shs[j * 3 + 1] = lerp(configData.chunkMinshs[1], configData.chunkMaxshs[1], shsData[1]);
					shs[i].shs[j * 3 + 2] = lerp(configData.chunkMinshs[2], configData.chunkMaxshs[2], shsData[2]);
				}
			}

			minn = minn.cwiseMin(configData.chunkMinpos);
			maxx = maxx.cwiseMax(configData.chunkMaxpos);

			SetModelProcess(fileMd5, 3 + chunkIdx * 1.0f / chunkCount * 97.0f);
		}
	}
	else if (quality == 1)
	{
		std::vector<HighPoint<15>> points(lcount);
		infile.read((char*)points.data(), lcount * sizeof(HighPoint<15>));
		SetModelProcess(fileMd5, 3);
		// Resize our SoA data
		pos.resize(lcount);
		shs.resize(lcount);
		scales.resize(lcount);
		rot.resize(lcount);
		opacities.resize(lcount);
		//
		int chunkCount = (points.size() + kChunkSize - 1) / kChunkSize;
		//不分块了，按总的来算，多线程就弄多个最大最小值，再一起比较，最终每个变量只取一组最大最小值-----(不分块的话，场景大了，容易失真)
		for (int chunkIdx = 0; chunkIdx < chunkCount; chunkIdx++)
		{
			int splatBegin = MIN(chunkIdx * kChunkSize, points.size());
			int splatEnd = MIN((chunkIdx + 1) * kChunkSize, points.size());
			// calculate data bounds inside the chunk
			MiddleCompressData configData = middleCompressData[chunkIdx];
			// Move data from AoS to SoA
			int SH_N = (D + 1) * (D + 1);
			for (int i = splatBegin; i < splatEnd; i++)
			{
				/*
				vector<float>& posData = DecodePacked_21_21_21(points[i].pos);
				pos[i][0] = lerp(configData.chunkMinpos[0], configData.chunkMaxpos[0], posData[0]);
				pos[i][1] = lerp(configData.chunkMinpos[1], configData.chunkMaxpos[1], posData[1]);
				pos[i][2] = lerp(configData.chunkMinpos[2], configData.chunkMaxpos[2], posData[2]);
				//pos[i][0] = lerp(configData.chunkMinpos[0], configData.chunkMaxpos[0], points[i].pos[0]);
				//pos[i][1] = lerp(configData.chunkMinpos[1], configData.chunkMaxpos[1], points[i].pos[1]);
				//pos[i][2] = lerp(configData.chunkMinpos[2], configData.chunkMaxpos[2], points[i].pos[2]);
				*/
				pos[i][0] = points[i].pos[0];
				pos[i][1] = points[i].pos[1];
				pos[i][2] = points[i].pos[2];

				vector<float>& scaleData = DecodePacked_21_21_21(points[i].scale);
				scales[i].scale[0] = lerp(configData.chunkMinscl[0], configData.chunkMaxscl[0], scaleData[0]);
				scales[i].scale[1] = lerp(configData.chunkMinscl[1], configData.chunkMaxscl[1], scaleData[1]);
				scales[i].scale[2] = lerp(configData.chunkMinscl[2], configData.chunkMaxscl[2], scaleData[2]);

				// int32
				vector<float>& rotData = DecodePacked_16_16_16_16(points[i].rot);
				rot[i].rot[0] = lerp(configData.chunkMinrot[0], configData.chunkMaxrot[0], rotData[0]);
				rot[i].rot[1] = lerp(configData.chunkMinrot[1], configData.chunkMaxrot[1], rotData[1]);
				rot[i].rot[2] = lerp(configData.chunkMinrot[2], configData.chunkMaxrot[2], rotData[2]);
				rot[i].rot[3] = lerp(configData.chunkMinrot[3], configData.chunkMaxrot[3], rotData[3]);
				/* int64
				vector<float> rotData = DecodePacked_16_16_16_16(points[i].rot);
				rot[i].rot[0] = lerp(configData.chunkMinrot[0], configData.chunkMaxrot[0], rotData[0]);
				rot[i].rot[1] = lerp(configData.chunkMinrot[1], configData.chunkMaxrot[1], rotData[1]);
				rot[i].rot[2] = lerp(configData.chunkMinrot[2], configData.chunkMaxrot[2], rotData[2]);
				rot[i].rot[3] = lerp(configData.chunkMinrot[3], configData.chunkMaxrot[3], rotData[3]);
				*/
				/* float4
				rot[i].rot[0] = lerp(configData.chunkMinrot[0], configData.chunkMaxrot[0], points[i].rot[0]);
				rot[i].rot[1] = lerp(configData.chunkMinrot[1], configData.chunkMaxrot[1], points[i].rot[1]);
				rot[i].rot[2] = lerp(configData.chunkMinrot[2], configData.chunkMaxrot[2], points[i].rot[2]);
				rot[i].rot[3] = lerp(configData.chunkMinrot[3], configData.chunkMaxrot[3], points[i].rot[3]);
				*/
				//color
							// int32
				vector<float>& colorData = DecodePacked_8_8_8_8(points[i].color);
				shs[i].shs[0] = lerp(configData.chunkMincol[0], configData.chunkMaxcol[0], colorData[0]);
				shs[i].shs[1] = lerp(configData.chunkMincol[1], configData.chunkMaxcol[1], colorData[1]);
				shs[i].shs[2] = lerp(configData.chunkMincol[2], configData.chunkMaxcol[2], colorData[2]);
				opacities[i] = lerp(configData.chunkMincol[3], configData.chunkMaxcol[3], colorData[3]);

				for (int j = 1; j < SH_N; j++)
				{
					vector<float> shsData = DecodePacked_11_10_11(points[i].shs[j - 1]);
					//vector<float>& shsData = DecodePacked_5_6_5(points[i].shs[j - 1]);

					shs[i].shs[j * 3 + 0] = lerp(configData.chunkMinshs[0], configData.chunkMaxshs[0], shsData[0]);
					shs[i].shs[j * 3 + 1] = lerp(configData.chunkMinshs[1], configData.chunkMaxshs[1], shsData[1]);
					shs[i].shs[j * 3 + 2] = lerp(configData.chunkMinshs[2], configData.chunkMaxshs[2], shsData[2]);
				}
			}

			minn = minn.cwiseMin(configData.chunkMinpos);
			maxx = maxx.cwiseMax(configData.chunkMaxpos);
			SetModelProcess(fileMd5, 3 + chunkIdx * 1.0f / chunkCount * 97.0f);
		}
	}
	
	return lcount;
}