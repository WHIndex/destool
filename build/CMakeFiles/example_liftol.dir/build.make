# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /var/lib/snapd/snap/cmake/1457/bin/cmake

# The command to remove a file.
RM = /var/lib/snapd/snap/cmake/1457/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wanghui/gre/liftol

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wanghui/gre/liftol/build

# Include any dependencies generated for this target.
include CMakeFiles/example_liftol.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/example_liftol.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/example_liftol.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/example_liftol.dir/flags.make

CMakeFiles/example_liftol.dir/codegen:
.PHONY : CMakeFiles/example_liftol.dir/codegen

CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.o: CMakeFiles/example_liftol.dir/flags.make
CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.o: /home/wanghui/gre/liftol/src/examples/example_liftol.cpp
CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.o: CMakeFiles/example_liftol.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/wanghui/gre/liftol/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.o"
	/opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.o -MF CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.o.d -o CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.o -c /home/wanghui/gre/liftol/src/examples/example_liftol.cpp

CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.i"
	/opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wanghui/gre/liftol/src/examples/example_liftol.cpp > CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.i

CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.s"
	/opt/rh/devtoolset-9/root/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wanghui/gre/liftol/src/examples/example_liftol.cpp -o CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.s

# Object files for target example_liftol
example_liftol_OBJECTS = \
"CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.o"

# External object files for target example_liftol
example_liftol_EXTERNAL_OBJECTS =

example_liftol: CMakeFiles/example_liftol.dir/src/examples/example_liftol.cpp.o
example_liftol: CMakeFiles/example_liftol.dir/build.make
example_liftol: /usr/lib64/libjemalloc.so
example_liftol: /opt/intel/mkl/lib/intel64/libmkl_intel_ilp64.a
example_liftol: /opt/intel/mkl/lib/intel64/libmkl_sequential.a
example_liftol: /opt/intel/mkl/lib/intel64/libmkl_core.a
example_liftol: /usr/local/lib64/libtbb.so
example_liftol: /opt/rh/devtoolset-9/root/usr/lib/gcc/x86_64-redhat-linux/9/libgomp.so
example_liftol: /lib64/libpthread.so
example_liftol: CMakeFiles/example_liftol.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/wanghui/gre/liftol/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example_liftol"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_liftol.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/example_liftol.dir/build: example_liftol
.PHONY : CMakeFiles/example_liftol.dir/build

CMakeFiles/example_liftol.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/example_liftol.dir/cmake_clean.cmake
.PHONY : CMakeFiles/example_liftol.dir/clean

CMakeFiles/example_liftol.dir/depend:
	cd /home/wanghui/gre/liftol/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wanghui/gre/liftol /home/wanghui/gre/liftol /home/wanghui/gre/liftol/build /home/wanghui/gre/liftol/build /home/wanghui/gre/liftol/build/CMakeFiles/example_liftol.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/example_liftol.dir/depend

