#ifndef NNLLMGGA_H
#define NNLLMGGA_H
#ifdef DFTFE_WITH_TORCH
#  include <string>
#  include <torch/torch.h>
#  include <excDensityPositivityCheckTypes.h>
namespace dftfe
{
  class NNLLMGGA
  {
  public:
    NNLLMGGA(std::string                          modelFilename,
             const bool                           isSpinPolarized = false,
             const excDensityPositivityCheckTypes densityPositivityCheckType =
               excDensityPositivityCheckTypes::MAKE_POSITIVE);
    ~NNLLMGGA();

    void
    evaluateexc(const double     *rho,
                const double     *sigma,
                const double     *laprho,
                const dftfe::uInt numPoints,
                double           *exc);
    void
    evaluatevxc(const double     *rho,
                const double     *sigma,
                const double     *laprho,
                const dftfe::uInt numPoints,
                double           *exc,
                double           *dexc);

  private:
    std::string                          d_modelFilename;
    std::string                          d_ptcFilename;
    torch::jit::script::Module          *d_model;
    const bool                           d_isSpinPolarized;
    double                               d_rhoTol;
    double                               d_sThreshold;
    const excDensityPositivityCheckTypes d_densityPositivityCheckType;
  };
} // namespace dftfe
#endif
#endif // NNLLMGGA_H
