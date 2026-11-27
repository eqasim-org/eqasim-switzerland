#include "io.hpp"
#if defined(SUMO_BUILD_INDEX) || !defined(SUMO_BUILD_SINGLE)
#include "index.hpp"
#endif
#if defined(SUMO_BUILD_REALIZE) || !defined(SUMO_BUILD_SINGLE)
#include "routes.hpp"
#endif
#if defined(SUMO_BUILD_CUT) || !defined(SUMO_BUILD_SINGLE)
#include "cut.hpp"
#endif
#if defined(SUMO_BUILD_EXPORT) || !defined(SUMO_BUILD_SINGLE)
#include "export.hpp"
#endif
int main(int argc, char **argv) {
    try {
        sumo_native::Options options(argc, argv);
        std::string command = argv[1];
#if defined(SUMO_BUILD_INDEX) || !defined(SUMO_BUILD_SINGLE)
        if (command == "index") {
            sumo_native::index_xml(options);
            return 0;
        }
#endif
#if defined(SUMO_BUILD_REALIZE) || !defined(SUMO_BUILD_SINGLE)
        if (command == "realize") {
            sumo_native::realize(options);
            return 0;
        }
#endif
#if defined(SUMO_BUILD_CUT) || !defined(SUMO_BUILD_SINGLE)
        if (command == "cut") {
            sumo_native::cut(options);
            return 0;
        }
#endif
#if defined(SUMO_BUILD_EXPORT) || !defined(SUMO_BUILD_SINGLE)
        if (command == "export") {
            sumo_native::export_xml(options);
            return 0;
        }
#endif
        throw std::runtime_error("Unsupported native command: " + command);
    } catch (const std::exception &error) {
        std::cerr << "SUMO native processing: " << error.what() << '\n';
        return 1;
    }
}
