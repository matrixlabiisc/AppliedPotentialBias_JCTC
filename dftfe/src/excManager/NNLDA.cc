#ifdef DFTFE_WITH_TORCH
#  include <torch/script.h>
#  include <NNLDA.h>
#  include <iostream>
#  include <vector>
#  include <algorithm>
#  include <iterator>
#  include <Exceptions.h>
namespace dftfe
{
  namespace
  {
    struct CastToFloat
    {
      float
      operator()(double value) const
      {
        return static_cast<float>(value);
      }
    };

    struct CastToDouble
    {
      double
      operator()(float value) const
      {
        return static_cast<double>(value);
      }
    };

    std::string
    trim(const std::string &str, const std::string &whitespace = " \t")
    {
      std::size_t strBegin = str.find_first_not_of(whitespace);
      if (strBegin == std::string::npos)
        return ""; // no content
      std::size_t strEnd   = str.find_last_not_of(whitespace);
      std::size_t strRange = strEnd - strBegin + 1;
      return str.substr(strBegin, strRange);
    }

    std::map<std::string, std::string>
    getKeyValuePairs(const std::string filename, const std::string delimiter)
    {
      std::map<std::string, std::string> returnValue;
      std::ifstream                      readFile;
      std::string                        readLine;
      readFile.open(filename.c_str());
      utils::throwException<utils::InvalidArgument>(readFile.is_open(),
                                                    "Unable to open file " +
                                                      filename);
      while (std::getline(readFile, readLine))
        {
          auto pos = readLine.find_last_of(delimiter);
          if (pos != std::string::npos)
            {
              std::string key   = trim(readLine.substr(0, pos));
              std::string value = trim(readLine.substr(pos + 1));
              returnValue[key]  = value;
            }
        }

      readFile.close();
      return returnValue;
    }

    void
    excSpinUnpolarized(
      const double                        *rho,
      const dftfe::uInt                    numPoints,
      double                              *exc,
      torch::jit::script::Module          *model,
      const excDensityPositivityCheckTypes densityPositivityCheckType,
      const double                         rhoTol)
    {
      std::vector<double> rhoModified(numPoints, 0.0);
      if (densityPositivityCheckType ==
          excDensityPositivityCheckTypes::EXCEPTION_POSITIVE)
        for (dftfe::uInt i = 0; i < numPoints; ++i)
          {
            std::string errMsg =
              "Negative electron-density encountered during xc evaluations";
            dftfe::utils::throwException(rho[i] > 0, errMsg);
          }
      else if (densityPositivityCheckType ==
               excDensityPositivityCheckTypes::MAKE_POSITIVE)
        for (dftfe::uInt i = 0; i < numPoints; ++i)
          {
            rhoModified[i] =
              std::max(rho[i], 0.0); // d_rhoTol will be added subsequently
          }
      else
        for (dftfe::uInt i = 0; i < numPoints; ++i)
          {
            rhoModified[i] = rho[i];
          }

      std::vector<float> rhoFloat(0);
      std::transform(&rhoModified[0],
                     &rhoModified[0] + numPoints,
                     std::back_inserter(rhoFloat),
                     CastToFloat());

      auto options =
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
      torch::Tensor rhoTensor =
        torch::from_blob(&rhoFloat[0], {numPoints, 1}, options).clone();
      rhoTensor += rhoTol;
      std::vector<torch::jit::IValue> input(0);
      input.push_back(rhoTensor);
      auto excTensor = model->forward(input).toTensor();
      for (dftfe::uInt i = 0; i < numPoints; ++i)
        exc[i] = static_cast<double>(excTensor[i][0].item<float>()) /
                 (rhoModified[i] + rhoTol);
    }

    void
    excSpinPolarized(
      const double                        *rho,
      const dftfe::uInt                    numPoints,
      double                              *exc,
      torch::jit::script::Module          *model,
      const excDensityPositivityCheckTypes densityPositivityCheckType,
      const double                         rhoTol)
    {
      std::vector<double> rhoModified(2 * numPoints, 0.0);
      if (densityPositivityCheckType ==
          excDensityPositivityCheckTypes::EXCEPTION_POSITIVE)
        for (dftfe::uInt i = 0; i < 2 * numPoints; ++i)
          {
            std::string errMsg =
              "Negative electron-density encountered during xc evaluations";
            dftfe::utils::throwException(rho[i] > 0, errMsg);
          }
      else if (densityPositivityCheckType ==
               excDensityPositivityCheckTypes::MAKE_POSITIVE)
        for (dftfe::uInt i = 0; i < 2 * numPoints; ++i)
          {
            rhoModified[i] =
              std::max(rho[i], 0.0); // d_rhoTol will be added subsequently
          }
      else
        for (dftfe::uInt i = 0; i < 2 * numPoints; ++i)
          {
            rhoModified[i] = rho[i];
          }

      std::vector<float> rhoFloat(0);
      std::transform(&rhoModified[0],
                     &rhoModified[0] + 2 * numPoints,
                     std::back_inserter(rhoFloat),
                     CastToFloat());

      auto options =
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
      torch::Tensor rhoTensor =
        torch::from_blob(&rhoFloat[0], {numPoints, 2}, options).clone();
      rhoTensor += rhoTol;
      std::vector<torch::jit::IValue> input(0);
      input.push_back(rhoTensor);
      auto excTensor = model->forward(input).toTensor();
      for (dftfe::uInt i = 0; i < numPoints; ++i)
        exc[i] = static_cast<double>(excTensor[i][0].item<float>()) /
                 (rhoModified[2 * i] + rhoModified[2 * i + 1] + 2 * rhoTol);
    }

    void
    vxcSpinUnpolarized(
      const double                        *rho,
      const dftfe::uInt                    numPoints,
      double                              *exc,
      double                              *vxc,
      torch::jit::script::Module          *model,
      const excDensityPositivityCheckTypes densityPositivityCheckType,
      const double                         rhoTol)
    {
      std::vector<double> rhoModified(numPoints, 0.0);
      if (densityPositivityCheckType ==
          excDensityPositivityCheckTypes::EXCEPTION_POSITIVE)
        for (dftfe::uInt i = 0; i < numPoints; ++i)
          {
            std::string errMsg =
              "Negative electron-density encountered during xc evaluations";
            dftfe::utils::throwException(rho[i] > 0, errMsg);
          }
      else if (densityPositivityCheckType ==
               excDensityPositivityCheckTypes::MAKE_POSITIVE)
        for (dftfe::uInt i = 0; i < numPoints; ++i)
          {
            rhoModified[i] =
              std::max(rho[i], 0.0); // d_rhoTol will be added subsequently
          }
      else
        for (dftfe::uInt i = 0; i < numPoints; ++i)
          {
            rhoModified[i] = rho[i];
          }

      std::vector<float> rhoFloat(0);
      std::transform(&rhoModified[0],
                     &rhoModified[0] + numPoints,
                     std::back_inserter(rhoFloat),
                     CastToFloat());

      auto options =
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
      torch::Tensor rhoTensor =
        torch::from_blob(&rhoFloat[0], {numPoints, 1}, options).clone();
      rhoTensor += rhoTol;
      std::vector<torch::jit::IValue> input(0);
      input.push_back(rhoTensor);
      auto excTensor   = model->forward(input).toTensor();
      auto grad_output = torch::ones_like(excTensor);
      auto vxcTensor   = torch::autograd::grad({excTensor},
                                               {rhoTensor},
                                             /*grad_outputs=*/{grad_output},
                                             /*create_graph=*/true)[0];
      for (dftfe::uInt i = 0; i < numPoints; ++i)
        {
          exc[i] = static_cast<double>(excTensor[i][0].item<float>()) /
                   (rhoModified[i] + rhoTol);
          vxc[i] = static_cast<double>(vxcTensor[i][0].item<float>());
        }
    }

    void
    vxcSpinPolarized(
      const double                        *rho,
      const dftfe::uInt                    numPoints,
      double                              *exc,
      double                              *vxc,
      torch::jit::script::Module          *model,
      const excDensityPositivityCheckTypes densityPositivityCheckType,
      const double                         rhoTol)
    {
      std::vector<double> rhoModified(2 * numPoints, 0.0);
      if (densityPositivityCheckType ==
          excDensityPositivityCheckTypes::EXCEPTION_POSITIVE)
        for (dftfe::uInt i = 0; i < 2 * numPoints; ++i)
          {
            std::string errMsg =
              "Negative electron-density encountered during xc evaluations";
            dftfe::utils::throwException(rho[i] > 0, errMsg);
          }
      else if (densityPositivityCheckType ==
               excDensityPositivityCheckTypes::MAKE_POSITIVE)
        for (dftfe::uInt i = 0; i < 2 * numPoints; ++i)
          {
            rhoModified[i] =
              std::max(rho[i], 0.0); // d_rhoTol will be added subsequently
          }
      else
        for (dftfe::uInt i = 0; i < 2 * numPoints; ++i)
          {
            rhoModified[i] = rho[i];
          }

      std::vector<float> rhoFloat(0);
      std::transform(&rhoModified[0],
                     &rhoModified[0] + 2 * numPoints,
                     std::back_inserter(rhoFloat),
                     CastToFloat());

      auto options =
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
      torch::Tensor rhoTensor =
        torch::from_blob(&rhoFloat[0], {numPoints, 2}, options).clone();
      rhoTensor += rhoTol;
      std::vector<torch::jit::IValue> input(0);
      input.push_back(rhoTensor);
      auto excTensor   = model->forward(input).toTensor();
      auto grad_output = torch::ones_like(excTensor);
      auto vxcTensor   = torch::autograd::grad({excTensor},
                                               {rhoTensor},
                                             /*grad_outputs=*/{grad_output},
                                             /*create_graph=*/true)[0];
      for (dftfe::uInt i = 0; i < numPoints; ++i)
        {
          exc[i] = static_cast<double>(excTensor[i][0].item<float>()) /
                   (rhoModified[2 * i] + rhoModified[2 * i + 1] + 2 * rhoTol);
          for (dftfe::uInt j = 0; j < 2; ++j)
            vxc[2 * i + j] = static_cast<double>(vxcTensor[i][j].item<float>());
        }
    }
  } // namespace

  NNLDA::NNLDA(std::string                          modelFilename,
               const bool                           isSpinPolarized /*=false*/,
               const excDensityPositivityCheckTypes densityPositivityCheckType)
    : d_modelFilename(modelFilename)
    , d_isSpinPolarized(isSpinPolarized)
    , d_densityPositivityCheckType(densityPositivityCheckType)
  {
    std::map<std::string, std::string> modelKeyValues =
      getKeyValuePairs(d_modelFilename, "=");

    std::vector<std::string> keysToFind = {"PTC_FILE", "RHO_TOL"};

    // check if all required keys are found
    for (dftfe::uInt i = 0; i < keysToFind.size(); ++i)
      {
        bool found = false;
        for (auto it = modelKeyValues.begin(); it != modelKeyValues.end(); ++it)
          {
            if (keysToFind[i] == it->first)
              found = true;
          }
        utils::throwException(found,
                              "Unable to find the key " + keysToFind[i] +
                                " in file " + d_modelFilename);
      }

    d_ptcFilename = modelKeyValues["PTC_FILE"];
    d_rhoTol      = std::stod(modelKeyValues["RHO_TOL"]);

    d_model  = new torch::jit::script::Module;
    *d_model = torch::jit::load(d_ptcFilename);
    // Explicitly load model onto CPU, you can use kGPU if you are on Linux
    // and have libtorch version with CUDA support (and a GPU)
    d_model->to(torch::kCPU);
  }

  NNLDA::~NNLDA()
  {
    delete d_model;
  }

  void
  NNLDA::evaluateexc(const double     *rho,
                     const dftfe::uInt numPoints,
                     double           *exc)
  {
    if (!d_isSpinPolarized)
      excSpinUnpolarized(
        rho, numPoints, exc, d_model, d_densityPositivityCheckType, d_rhoTol);
    else
      excSpinPolarized(
        rho, numPoints, exc, d_model, d_densityPositivityCheckType, d_rhoTol);


    //  std::vector<float> rhoFloat(0);
    //  std::transform(rho, rho+numPoints,
    //	  std::back_inserter(rhoFloat),
    //	  CastToFloat());

    //  auto options =
    //  torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
    //  torch::Tensor rhoTensor = torch::from_blob(&rhoFloat[0], {numPoints,1},
    //  options).clone(); rhoTensor += d_rhoTol;
    //  std::vector<torch::jit::IValue> input(0);
    //  input.push_back(rhoTensor);
    //  auto excTensor = d_model->forward(input).toTensor();
    //  for(dftfe::uInt i = 0; i < numPoints; ++i)
    //	exc[i] = static_cast<double>(excTensor[i][0].item<float>());
  }

  void
  NNLDA::evaluatevxc(const double     *rho,
                     const dftfe::uInt numPoints,
                     double           *exc,
                     double           *vxc)
  {
    if (!d_isSpinPolarized)
      vxcSpinUnpolarized(rho,
                         numPoints,
                         exc,
                         vxc,
                         d_model,
                         d_densityPositivityCheckType,
                         d_rhoTol);
    else
      vxcSpinPolarized(rho,
                       numPoints,
                       exc,
                       vxc,
                       d_model,
                       d_densityPositivityCheckType,
                       d_rhoTol);

    // std::vector<float> rhoFloat(0);
    // std::transform(rho, rho+numPoints,
    //    std::back_inserter(rhoFloat),
    //    CastToFloat());

    // auto options =
    // torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
    // torch::Tensor rhoTensor = torch::from_blob(&rhoFloat[0], {numPoints,1},
    // options).clone(); rhoTensor += 1e-8; std::vector<torch::jit::IValue>
    // input(0); input.push_back(rhoTensor); auto excTensor =
    // d_model->forward(input).toTensor(); auto grad_output =
    // torch::ones_like(excTensor); auto vxcTensor =
    // torch::autograd::grad({excTensor}, {rhoTensor},
    // /*grad_outputs=*/{grad_output}, /*create_graph=*/true)[0]; for(unsigned
    // dftfe::Int i = 0; i < numPoints; ++i)
    //{
    //  exc[i] = static_cast<double>(excTensor[i][0].item<float>());
    //  vxc[i] = static_cast<double>(vxcTensor[i][0].item<float>());
    //}
  }

} // namespace dftfe
#endif
