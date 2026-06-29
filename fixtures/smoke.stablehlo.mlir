module @jit_smoke attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8x64xf32>, %arg1: tensor<64x64xf32>) -> (tensor<8x64xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x64xf32>) -> tensor<8x64xf32>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %3 = stablehlo.maximum %2, %1 : tensor<8xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %6 = stablehlo.subtract %0, %5 : tensor<8x64xf32>
    %7 = stablehlo.exponential %6 : tensor<8x64xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %11 = stablehlo.divide %7, %10 : tensor<8x64xf32>
    return %11 : tensor<8x64xf32>
  }
}
