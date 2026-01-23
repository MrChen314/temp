Using pip 25.1.1 from /usr/local/lib/python3.12/dist-packages/pip (python 3.12)
Looking in indexes: http://mirrors.baidubce.com/pypi/simple/
Obtaining file:///home/users/chenquanlin/workspace/dsv32/FlashMLA
  Preparing metadata (setup.py): started
  Running command python setup.py egg_info
  Compiling using NVCC 12.9
  running egg_info
  creating /tmp/pip-pip-egg-info-v96tnlu7/flash_mla.egg-info
  writing /tmp/pip-pip-egg-info-v96tnlu7/flash_mla.egg-info/PKG-INFO
  writing dependency_links to /tmp/pip-pip-egg-info-v96tnlu7/flash_mla.egg-info/dependency_links.txt
  writing top-level names to /tmp/pip-pip-egg-info-v96tnlu7/flash_mla.egg-info/top_level.txt
  writing manifest file '/tmp/pip-pip-egg-info-v96tnlu7/flash_mla.egg-info/SOURCES.txt'
  reading manifest file '/tmp/pip-pip-egg-info-v96tnlu7/flash_mla.egg-info/SOURCES.txt'
  adding license file 'LICENSE'
  writing manifest file '/tmp/pip-pip-egg-info-v96tnlu7/flash_mla.egg-info/SOURCES.txt'
  Preparing metadata (setup.py): finished with status 'done'
Installing collected packages: flash_mla
  DEPRECATION: Legacy editable install of flash_mla==1.0.0+3ddd21e from file:///home/users/chenquanlin/workspace/dsv32/FlashMLA (setup.py develop) is deprecated. pip 25.3 will enforce this behaviour change. A possible replacement is to add a pyproject.toml or enable --use-pep517, and use setuptools >= 64. If the resulting installation is not behaving as expected, try using --config-settings editable_mode=compat. Please consult the setuptools documentation for more information. Discussion can be found at https://github.com/pypa/pip/issues/11457
  Running setup.py develop for flash_mla
    Running command python setup.py develop
    Compiling using NVCC 12.9
    running develop
    /usr/local/lib/python3.12/dist-packages/setuptools/command/develop.py:41: EasyInstallDeprecationWarning: easy_install command is deprecated.
    !!

            ********************************************************************************
            Please avoid running ``setup.py`` and ``easy_install``.
            Instead, use pypa/build, pypa/installer or other
            standards-based tools.

            See https://github.com/pypa/setuptools/issues/917 for details.
            ********************************************************************************

    !!
      easy_install.initialize_options(self)
    /usr/local/lib/python3.12/dist-packages/setuptools/_distutils/cmd.py:90: SetuptoolsDeprecationWarning: setup.py install is deprecated.
    !!

            ********************************************************************************
            Please avoid running ``setup.py`` directly.
            Instead, use pypa/build, pypa/installer or other
            standards-based tools.

            See https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html for details.
            ********************************************************************************

    !!
      self.initialize_options()
    running egg_info
    writing flash_mla.egg-info/PKG-INFO
    writing dependency_links to flash_mla.egg-info/dependency_links.txt
    writing top-level names to flash_mla.egg-info/top_level.txt
    reading manifest file 'flash_mla.egg-info/SOURCES.txt'
    adding license file 'LICENSE'
    writing manifest file 'flash_mla.egg-info/SOURCES.txt'
    running build_ext
    building 'flash_mla.cuda' extension
    Emitting ninja build file /home/users/chenquanlin/workspace/dsv32/FlashMLA/build/temp.linux-x86_64-cpython-312/build.ninja...
    Compiling objects...
    Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
    [1/1] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output /home/users/chenquanlin/workspace/dsv32/FlashMLA/build/temp.linux-x86_64-cpython-312/csrc/sm100/prefill/sparse/fwd.o.d -I/home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc -I/home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/sm90 -I/home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/cutlass/include -I/home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/cutlass/tools/util/include -I/usr/local/lib/python3.12/dist-packages/torch/include -I/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda/include -I/usr/include/python3.12 -c -c /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/sm100/prefill/sparse/fwd.cu -o /home/users/chenquanlin/workspace/dsv32/FlashMLA/build/temp.linux-x86_64-cpython-312/csrc/sm100/prefill/sparse/fwd.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O3 -std=c++17 -DNDEBUG -D_USE_MATH_DEFINES -Wno-deprecated-declarations -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --ptxas-options=-v,--register-usage-level=10 -gencode arch=compute_100a,code=sm_100a -gencode arch=compute_90a,code=sm_90a --threads 32 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1016"' -DTORCH_EXTENSION_NAME=cuda
    FAILED: /home/users/chenquanlin/workspace/dsv32/FlashMLA/build/temp.linux-x86_64-cpython-312/csrc/sm100/prefill/sparse/fwd.o
    /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output /home/users/chenquanlin/workspace/dsv32/FlashMLA/build/temp.linux-x86_64-cpython-312/csrc/sm100/prefill/sparse/fwd.o.d -I/home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc -I/home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/sm90 -I/home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/cutlass/include -I/home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/cutlass/tools/util/include -I/usr/local/lib/python3.12/dist-packages/torch/include -I/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda/include -I/usr/include/python3.12 -c -c /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/sm100/prefill/sparse/fwd.cu -o /home/users/chenquanlin/workspace/dsv32/FlashMLA/build/temp.linux-x86_64-cpython-312/csrc/sm100/prefill/sparse/fwd.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O3 -std=c++17 -DNDEBUG -D_USE_MATH_DEFINES -Wno-deprecated-declarations -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math --ptxas-options=-v,--register-usage-level=10 -gencode arch=compute_100a,code=sm_100a -gencode arch=compute_90a,code=sm_90a --threads 32 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1016"' -DTORCH_EXTENSION_NAME=cuda
    /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/sm100/prefill/sparse/fwd.cu(830): error: name followed by "::" must be a class or namespace name
              Layout<Shape<Int<Kernel::B_H/2>, Int<Kernel::B_TOPK>>, Stride<Int<Kernel::B_TOPK>, _1>>{}
                               ^

    /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/sm100/prefill/sparse/fwd.cu(830): error: name followed by "::" must be a class or namespace name
              Layout<Shape<Int<Kernel::B_H/2>, Int<Kernel::B_TOPK>>, Stride<Int<Kernel::B_TOPK>, _1>>{}
                                                   ^

    /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/sm100/prefill/sparse/fwd.cu(830): error: name followed by "::" must be a class or namespace name
              Layout<Shape<Int<Kernel::B_H/2>, Int<Kernel::B_TOPK>>, Stride<Int<Kernel::B_TOPK>, _1>>{}
                                                                                ^

    /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/cutlass/include/cute/container/tuple.hpp(199): error: incomplete type "cute::eso::ESO<true, true, cute::Int<<error-constant>>, cute::Int<<error-constant>>>" (aka "cute::eso::ESO<true, true, cute::C<<error-constant>>, cute::C<<error-constant>>>") is not allowed
      struct tuple : eso::ESO_t<T...>
                     ^
              detected during:
                instantiation of class "cute::tuple<T...> [with T=<cute::C<<error-constant>>, cute::C<<error-constant>>>]" at line 843 of /usr/include/c++/13/type_traits
                instantiation of class "std::is_empty<_Tp> [with _Tp=cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>]" at line 82
                instantiation of "const bool cute::eso::is_first_empty_v [with First=cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, Rest=<cute::tuple<cute::C<<error-constant>>, cute::_1>>]" at line 87
                instantiation of type "cute::eso::ESO_t<cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, cute::tuple<cute::C<<error-constant>>, cute::_1>>" at line 199
                instantiation of class "cute::tuple<T...> [with T=<cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, cute::tuple<cute::C<<error-constant>>, cute::_1>>]" at line 100 of /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/cutlass/include/cute/layout.hpp
                instantiation of class "cute::Layout<Shape, Stride> [with Shape=cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, Stride=cute::tuple<cute::C<<error-constant>>, cute::_1>]" at line 830 of /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/sm100/prefill/sparse/fwd.cu

    /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/cutlass/include/cute/container/tuple.hpp(199): error: incomplete type "cute::eso::ESO<true, true, cute::Int<<error-constant>>, cute::_1>" (aka "cute::eso::ESO<true, true, cute::C<<error-constant>>, cute::C<1>>") is not allowed
      struct tuple : eso::ESO_t<T...>
                     ^
              detected during:
                instantiation of class "cute::tuple<T...> [with T=<cute::C<<error-constant>>, cute::_1>]" at line 843 of /usr/include/c++/13/type_traits
                instantiation of class "std::is_empty<_Tp> [with _Tp=cute::tuple<cute::C<<error-constant>>, cute::_1>]" at line 84
                instantiation of "const bool cute::eso::is_rest_empty_v [with First=cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, Rest=<cute::tuple<cute::C<<error-constant>>, cute::_1>>]" at line 87
                instantiation of type "cute::eso::ESO_t<cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, cute::tuple<cute::C<<error-constant>>, cute::_1>>" at line 199
                instantiation of class "cute::tuple<T...> [with T=<cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, cute::tuple<cute::C<<error-constant>>, cute::_1>>]" at line 100 of /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/cutlass/include/cute/layout.hpp
                instantiation of class "cute::Layout<Shape, Stride> [with Shape=cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, Stride=cute::tuple<cute::C<<error-constant>>, cute::_1>]" at line 830 of /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/sm100/prefill/sparse/fwd.cu

    /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/cutlass/include/cute/container/tuple.hpp(199): error: incomplete type "cute::eso::ESO<true, true, cute::Shape<cute::Int<<error-constant>>, cute::Int<<error-constant>>>, cute::Stride<cute::Int<<error-constant>>, cute::_1>>" (aka "cute::eso::ESO<true, true, cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, cute::tuple<cute::C<<error-constant>>, cute::C<1>>>") is not allowed
      struct tuple : eso::ESO_t<T...>
                     ^
              detected during:
                instantiation of class "cute::tuple<T...> [with T=<cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, cute::tuple<cute::C<<error-constant>>, cute::_1>>]" at line 100 of /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/cutlass/include/cute/layout.hpp
                instantiation of class "cute::Layout<Shape, Stride> [with Shape=cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, Stride=cute::tuple<cute::C<<error-constant>>, cute::_1>]" at line 830 of /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/sm100/prefill/sparse/fwd.cu

    /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/cutlass/include/cute/container/tuple.hpp(205): error: "ESO_t" is not a nonstatic data member or base class of class "cute::tuple<cute::Shape<cute::Int<<error-constant>>, cute::Int<<error-constant>>>, cute::Stride<cute::Int<<error-constant>>, cute::_1>>" (aka "cute::tuple<cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, cute::tuple<cute::C<<error-constant>>, cute::C<1>>>")
        tuple(T const&... t) : eso::ESO_t<T...>(t...) {}
                               ^
              detected during:
                instantiation of "cute::tuple<T...>::tuple(const T &...) [with T=<cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, cute::tuple<cute::C<<error-constant>>, cute::_1>>]" at line 109 of /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/cutlass/include/cute/layout.hpp
                instantiation of "cute::Layout<Shape, Stride>::Layout(const Shape &, const Stride &) [with Shape=cute::tuple<cute::C<<error-constant>>, cute::C<<error-constant>>>, Stride=cute::tuple<cute::C<<error-constant>>, cute::_1>]" at line 831 of /home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/sm100/prefill/sparse/fwd.cu

    7 errors detected in the compilation of "/home/users/chenquanlin/workspace/dsv32/FlashMLA/csrc/sm100/prefill/sparse/fwd.cu".
    ninja: build stopped: subcommand failed.
    Traceback (most recent call last):
      File "/usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py", line 2557, in _run_ninja_build
        subprocess.run(
      File "/usr/lib/python3.12/subprocess.py", line 571, in run
        raise CalledProcessError(retcode, process.args,
    subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "<string>", line 2, in <module>
      File "<pip-setuptools-caller>", line 35, in <module>
      File "/home/users/chenquanlin/workspace/dsv32/FlashMLA/setup.py", line 113, in <module>
        setup(
      File "/usr/local/lib/python3.12/dist-packages/setuptools/__init__.py", line 117, in setup
        return distutils.core.setup(**attrs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.12/dist-packages/setuptools/_distutils/core.py", line 186, in setup
        return run_commands(dist)
               ^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.12/dist-packages/setuptools/_distutils/core.py", line 202, in run_commands
        dist.run_commands()
      File "/usr/local/lib/python3.12/dist-packages/setuptools/_distutils/dist.py", line 1002, in run_commands
        self.run_command(cmd)
      File "/usr/local/lib/python3.12/dist-packages/setuptools/dist.py", line 1104, in run_command
        super().run_command(command)
      File "/usr/local/lib/python3.12/dist-packages/setuptools/_distutils/dist.py", line 1021, in run_command
        cmd_obj.run()
      File "/usr/local/lib/python3.12/dist-packages/setuptools/command/develop.py", line 35, in run
        self.install_for_development()
      File "/usr/local/lib/python3.12/dist-packages/setuptools/command/develop.py", line 112, in install_for_development
        self.run_command('build_ext')
      File "/usr/local/lib/python3.12/dist-packages/setuptools/_distutils/cmd.py", line 357, in run_command
        self.distribution.run_command(command)
      File "/usr/local/lib/python3.12/dist-packages/setuptools/dist.py", line 1104, in run_command
        super().run_command(command)
      File "/usr/local/lib/python3.12/dist-packages/setuptools/_distutils/dist.py", line 1021, in run_command
        cmd_obj.run()
      File "/usr/local/lib/python3.12/dist-packages/setuptools/command/build_ext.py", line 99, in run
        _build_ext.run(self)
      File "/usr/local/lib/python3.12/dist-packages/setuptools/_distutils/command/build_ext.py", line 368, in run
        self.build_extensions()
      File "/usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py", line 1052, in build_extensions
        build_ext.build_extensions(self)
      File "/usr/local/lib/python3.12/dist-packages/setuptools/_distutils/command/build_ext.py", line 484, in build_extensions
        self._build_extensions_serial()
      File "/usr/local/lib/python3.12/dist-packages/setuptools/_distutils/command/build_ext.py", line 510, in _build_extensions_serial
        self.build_extension(ext)
      File "/usr/local/lib/python3.12/dist-packages/setuptools/command/build_ext.py", line 264, in build_extension
        _build_ext.build_extension(self, ext)
      File "/usr/local/lib/python3.12/dist-packages/Cython/Distutils/build_ext.py", line 136, in build_extension
        super().build_extension(ext)
      File "/usr/local/lib/python3.12/dist-packages/setuptools/_distutils/command/build_ext.py", line 565, in build_extension
        objects = self.compiler.compile(
                  ^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py", line 836, in unix_wrap_ninja_compile
        _write_ninja_file_and_compile_objects(
      File "/usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py", line 2209, in _write_ninja_file_and_compile_objects
        _run_ninja_build(
      File "/usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py", line 2574, in _run_ninja_build
        raise RuntimeError(message) from e
    RuntimeError: Error compiling objects for extension
    error: subprocess-exited-with-error
    
    × python setup.py develop did not run successfully.
    │ exit code: 1
    ╰─> See above for output.
    
    note: This error originates from a subprocess, and is likely not a problem with pip.
    full command: /usr/bin/python -c '
    exec(compile('"'"''"'"''"'"'
    # This is <pip-setuptools-caller> -- a caller that pip uses to run setup.py
    #
    # - It imports setuptools before invoking setup.py, to enable projects that directly
    #   import from `distutils.core` to work with newer packaging standards.
    # - It provides a clear error message when setuptools is not installed.
    # - It sets `sys.argv[0]` to the underlying `setup.py`, when invoking `setup.py` so
    #   setuptools doesn'"'"'t think the script is `-c`. This avoids the following warning:
    #     manifest_maker: standard file '"'"'-c'"'"' not found".
    # - It generates a shim setup.py, for handling setup.cfg-only projects.
    import os, sys, tokenize, traceback
    
    try:
        import setuptools
    except ImportError:
        print(
            "ERROR: Can not execute `setup.py` since setuptools failed to import in "
            "the build environment with exception:",
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(1)
    
    __file__ = %r
    sys.argv[0] = __file__
    
    if os.path.exists(__file__):
        filename = __file__
        with tokenize.open(__file__) as f:
            setup_py_code = f.read()
    else:
        filename = "<auto-generated setuptools caller>"
        setup_py_code = "from setuptools import setup; setup()"
    
    exec(compile(setup_py_code, filename, "exec"))
    '"'"''"'"''"'"' % ('"'"'/home/users/chenquanlin/workspace/dsv32/FlashMLA/setup.py'"'"',), "<pip-setuptools-caller>", "exec"))' develop --no-deps
    cwd: /home/users/chenquanlin/workspace/dsv32/FlashMLA/
error: subprocess-exited-with-error

× python setup.py develop did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
