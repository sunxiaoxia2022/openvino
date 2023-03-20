# Build Plugin Using CMake {#openvino_docs_ie_plugin_dg_plugin_build}

OpenVINO build infrastructure provides the OpenVINO Developer Package for plugin development.

OpenVINO Developer Package
------------------------

To automatically generate the OpenVINO Developer Package, run the `cmake` tool during a OpenVINO build:

```bash
$ mkdir openvino-release-build
$ cd openvino-release-build
$ cmake -DCMAKE_BUILD_TYPE=Release ../openvino
```

Once the commands above are executed, the OpenVINO Developer Package is generated in the `openvino-release-build` folder. It consists of several files:
 - `OpenVINODeveloperPackageConfig.cmake` - the main CMake script which imports targets and provides compilation flags and CMake options.
 - `OpenVINODeveloperPackageConfig-version.cmake` - a file with a package version.
 - `targets_developer.cmake` - an automatically generated file which contains all targets exported from the OpenVINO build tree. This file is included by `OpenVINODeveloperPackageConfig.cmake` to import the following targets:
   - Libraries for plugin development:
       * `openvino::runtime` - shared OpenVINO library
       * `openvino::runtime::dev` - interface library with OpenVINO Developer API
       * `openvino::pugixml` - static Pugixml library
       * `openvino::xbyak` - interface library with Xbyak headers
       * `openvino::itt` - static library with tools for performance measurement using Intel ITT
   - Libraries for tests development:
       * `openvino::gtest`, `openvino::gtest_main`, `openvino::gmock` - Google Tests framework libraries
       * `openvino::commonTestUtils` - static library with common tests utilities 
       * `openvino::funcTestUtils` - static library with functional tests utilities 
       * `openvino::unitTestUtils` - static library with unit tests utilities 
       * `openvino::ngraphFunctions` - static library with the set of `ov::Model` builders
       * `openvino::funcSharedTests` - static library with common functional tests

> **NOTE**: it's enough just to run `cmake --build . --target ov_dev_targets` command to build only targets from the
> OpenVINO Developer package.

Build Plugin using OpenVINO Developer Package
------------------------

To build a plugin source tree using the OpenVINO Developer Package, run the commands below:

```cmake
$ mkdir template-plugin-release-build
$ cd template-plugin-release-build
$ cmake -DOpenVINODeveloperPackage_DIR=../openvino-release-build ../template-plugin
```

A common plugin consists of the following components:

1. Plugin code in the `src` folder
2. Code of tests in the `tests` folder

To build a plugin and its tests, run the following CMake scripts:

- Root `CMakeLists.txt`, which finds the OpenVINO Developer Package using the `find_package` CMake command and adds the `src` and `tests` subdirectories with plugin sources and their tests respectively:
@snippet template/CMakeLists.txt cmake:main
   > **NOTE**: The default values of the `ENABLE_TESTS`, `ENABLE_FUNCTIONAL_TESTS` options are shared via the OpenVINO Developer Package and they are the same as for the main OpenVINO build tree. You can override them during plugin build using the command below:
```bash
$ cmake -DENABLE_FUNCTIONAL_TESTS=OFF -DOpenVINODeveloperPackage_DIR=../openvino-release-build ../template-plugin
``` 

- `src/CMakeLists.txt` to build a plugin shared library from sources:
@snippet template/src/CMakeLists.txt cmake:plugin
   > **NOTE**: `openvino::runtime` target is imported from the OpenVINO Developer Package.

- `tests/functional/CMakeLists.txt` to build a set of functional plugin tests:
@snippet template/tests/functional/CMakeLists.txt cmake:functional_tests
   > **NOTE**: The `openvino::funcSharedTests` static library with common functional OpenVINO Plugin tests is imported via the OpenVINO Developer Package.
