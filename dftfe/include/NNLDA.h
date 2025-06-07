#ifndef NNLDA_H
#define NNLDA_H
#ifdef DFTFE_WITH_TORCH
#  include <string>
#  include <torch/torch.h>
#  include <excDensityPositivityCheckTypes.h>
namespace dftfe
{
  class NNLDA
  {
  public:
    NNLDA(std::string                          modelFilename,
          const bool                           isSpinPolarized = false,
          const excDensityPositivityCheckTypes densityPositivityCheckType =
            excDensityPositivityCheckTypes::MAKE_POSITIVE);

    ~NNLDA();
    void
    evaluateexc(const double *rho, const dftfe::uInt numPoints, double *exc);
    void
    evaluatevxc(const double     *rho,
                const dftfe::uInt numPoints,
                double           *exc,
                double           *vxc);

  private:
    std::string                          d_modelFilename;
    std::string                          d_ptcFilename;
    torch::jit::script::Module          *d_model;
    const bool                           d_isSpinPolarized;
    double                               d_rhoTol;
    const excDensityPositivityCheckTypes d_densityPositivityCheckType;
  };
} // namespace dftfe
#endif
#endif // NNLDA_H
