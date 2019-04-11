/* Parallel reading test of fits files */

#include <iostream>
#include <thread>
#include <vector>
#include <string>

#include "lsst/afw/table/Source.h"

// List of files to read by thread1
const std::vector<std::string> fileNames1{
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-000.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-001.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-002.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-003.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-004.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-005.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-006.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-007.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-008.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-009.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-010.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-011.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-012.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-013.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-014.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-015.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-016.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-017.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-018.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-019.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-020.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-021.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-022.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-023.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-024.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-025.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-026.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-027.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-028.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-029.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-030.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-031.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-032.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-033.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-034.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-035.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-036.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-037.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-038.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-039.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-040.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-041.fits"};

// List of files to read by thread2
const std::vector<std::string> fileNames2{
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-042.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-043.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-044.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-045.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-046.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-047.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-048.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-049.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-050.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-051.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-052.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-053.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-054.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-055.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-056.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-057.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-058.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-059.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-060.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-061.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-062.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-063.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-064.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-065.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-066.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-067.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-068.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-069.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-070.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-071.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-072.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-073.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-074.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-075.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-076.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-077.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-078.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-079.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-080.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-081.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-082.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-083.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-084.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-085.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-086.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-087.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-088.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-089.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-090.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-091.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-092.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-093.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-094.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-095.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-096.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-097.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-098.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-099.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-100.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-101.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-102.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0903982-103.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-000.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-001.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-002.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-003.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-004.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-005.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-006.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-007.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-008.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-009.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-010.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-011.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-012.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-013.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-014.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-015.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-016.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-017.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-018.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-019.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-020.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-021.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-022.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-023.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-024.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-025.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-026.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-027.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-028.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-029.fits",
        "validation_data_hsc/data/00671/HSC-I/output/SRC-0904006-030.fits"};

// This function is the entry point for the threads

void jobFunc(const std::vector<std::string>& fnames, int tid) {
    for (const auto& f : fnames) {
        using namespace lsst::afw::table;
        std::cout << "Thread " << tid << " file " << f << std::endl;
        auto readVector = SourceCatalog::readFits(f);
    }
};

int main() {
    std::thread t1(jobFunc, std::cref(fileNames1), 1);
    std::thread t2(jobFunc, std::cref(fileNames2), 2);
    t1.join();
    t2.join();

    return 0;
}
