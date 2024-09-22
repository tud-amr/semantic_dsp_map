
# Trouble Shooting
## Memory Problem

Problem
```
CMakeFiles/mapping.dir/src/mapping.cpp.o: in function `__tcf_0':
mapping.cpp:(.text+0x9): relocation truncated to fit: R_X86_64_PC32 against `.bss'
/usr/bin/ld: failed to convert GOTPCREL relocation; relink with --no-relax
```
This occurs usually when the map size in ```settings.h``` is too big for your PC's memory. Reduce the size by decrease one or more of the following parameters:
```
constexpr uint8_t C_VOXEL_NUM_AXIS_X_N = 8;
constexpr uint8_t C_VOXEL_NUM_AXIS_Y_N = 8;
constexpr uint8_t C_VOXEL_NUM_AXIS_Z_N = 7;

constexpr uint8_t C_MAX_PARTICLE_NUM_PER_VOXEL_N = 2; 
```

## Libusb problem when Nuitrack was installed

Problem 
```
/usr/lib/x86_64-linux-gnu/libpcl_io.so: undefined reference to `libusb_set_option'
```

This problem shows because Nuitrack is installed and PCL is linked to an old version libusb given by Nuitrack.

First, make sure libusb is installed by running
```
sudo apt install libusb-1.0-0-dev
```

Then check the symbolic link that libpcl_io.so uses by
```
ldd /usr/lib/x86_64-linux-gnu/libpcl_io.so
```

You will see ```libusb-1.0.so.0 => /usr/local/lib/nuitrack/libusb-1.0.so.0 (0x00007f5c3fbaf000)```. Further check by
```
readlink /usr/local/lib/nuitrack/libusb-1.0.so.0
```

You will see that ```libusb-1.0.so.0.1.0```, which is located in ```/usr/local/lib/nuitrack/```. The required version should be ```libusb-1.0.so.0.2.0```. Thus remake the link.
```
sudo cp /usr/lib/x86_64-linux-gnu/libusb-1.0.so.0.2.0 /usr/local/lib/nuitrack/ 
sudo rm /usr/local/lib/nuitrack/libusb-1.0.so.0
sudo ln -s /usr/local/lib/nuitrack/libusb-1.0.so.0.2.0 /usr/local/lib/nuitrack/libusb-1.0.so.0
```

This will fix the problem but nuitrack can possibly not be used. Reinstall nuitrack after ROS + PCL has been installed may be a way to use both of them. 

