#include <vector_types.h>
#include <optix_device.h>
#include "optixHello.h"
#include "helpers.h"
#include "random.h"




extern "C" {
	__constant__ Params params;
}

__host__ void test() {

	for (int i = 0; i < 500;i++) {
		params.frame_buffer[i] = make_color(make_float4(0,0,0,0));
		params.accum_buffer[i] = make_float4(0, 0, 0, 0);
	}


}
