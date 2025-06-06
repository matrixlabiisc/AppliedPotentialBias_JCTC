
#ifndef DFTFE_QuadDataCompositeWrite_H
#define DFTFE_QuadDataCompositeWrite_H

#include "CompositeData.h"
#include <vector>
#include <TypeConfig.h>

namespace dftfe
{
  namespace dftUtils
  {
    class QuadDataCompositeWrite : public CompositeData
    {
    public:
      QuadDataCompositeWrite(const std::vector<double> &vals);

      virtual void
      getCharArray(char *data) override;

      virtual void
      getMPIDataType(MPI_Datatype *mpi_datatype) override;

      virtual dftfe::Int
      getNumberCharsPerCompositeData() override;

    private:
      dftfe::uInt         d_charspernum;
      std::vector<double> d_vals;
    };
  } // namespace dftUtils
} // namespace dftfe
#endif // DFTFE_QuadDataCompositeWrite_H
