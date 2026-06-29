module @jit_attention_block attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8x64xf32>, %arg1: tensor<64x192xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x256xf32>, %arg4: tensor<256x64xf32>) -> (tensor<8x64xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %cst_0 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %3 = stablehlo.divide %1, %2 : tensor<8x1xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<8x64xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<8x64xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %cst_2 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %10 = stablehlo.divide %8, %9 : tensor<8x1xf32>
    %11 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %12 = stablehlo.subtract %arg0, %11 : tensor<8x64xf32>
    %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %14 = stablehlo.add %10, %13 : tensor<8x1xf32>
    %15 = stablehlo.sqrt %14 : tensor<8x1xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %17 = stablehlo.divide %12, %16 : tensor<8x64xf32>
    %18 = stablehlo.dot_general %17, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x192xf32>) -> tensor<8x192xf32>
    %19 = stablehlo.slice %18 [0:8, 0:64] : (tensor<8x192xf32>) -> tensor<8x64xf32>
    %20 = stablehlo.slice %18 [0:8, 64:128] : (tensor<8x192xf32>) -> tensor<8x64xf32>
    %21 = stablehlo.slice %18 [0:8, 128:192] : (tensor<8x192xf32>) -> tensor<8x64xf32>
    %22 = stablehlo.transpose %20, dims = [1, 0] : (tensor<8x64xf32>) -> tensor<64x8xf32>
    %23 = stablehlo.dot_general %19, %22, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x8xf32>) -> tensor<8x8xf32>
    %cst_4 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %24 = stablehlo.sqrt %cst_4 : tensor<f32>
    %25 = stablehlo.convert %24 : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
    %27 = stablehlo.divide %23, %26 : tensor<8x8xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %28 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
    %29 = call @tril(%28) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %30 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
    %31 = stablehlo.compare GT, %29, %30, FLOAT : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xi1>
    %cst_7 = stablehlo.constant dense<-1.000000e+09> : tensor<f32>
    %32 = call @_where(%31, %27, %cst_7) : (tensor<8x8xi1>, tensor<8x8xf32>, tensor<f32>) -> tensor<8x8xf32>
    %cst_8 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %33 = stablehlo.reduce(%32 init: %cst_8) applies stablehlo.maximum across dimensions = [1] : (tensor<8x8xf32>, tensor<f32>) -> tensor<8xf32>
    %cst_9 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %35 = stablehlo.maximum %34, %33 : tensor<8xf32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x8xf32>
    %38 = stablehlo.subtract %32, %37 : tensor<8x8xf32>
    %39 = stablehlo.exponential %38 : tensor<8x8xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %40 = stablehlo.reduce(%39 init: %cst_10) applies stablehlo.add across dimensions = [1] : (tensor<8x8xf32>, tensor<f32>) -> tensor<8xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x8xf32>
    %43 = stablehlo.divide %39, %42 : tensor<8x8xf32>
    %44 = stablehlo.dot_general %43, %21, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x8xf32>, tensor<8x64xf32>) -> tensor<8x64xf32>
    %45 = stablehlo.dot_general %44, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x64xf32>) -> tensor<8x64xf32>
    %46 = stablehlo.add %arg0, %45 : tensor<8x64xf32>
    %47 = stablehlo.dot_general %46, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x256xf32>) -> tensor<8x256xf32>
    %48 = stablehlo.multiply %47, %47 : tensor<8x256xf32>
    %49 = stablehlo.multiply %48, %47 : tensor<8x256xf32>
    %cst_11 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %50 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
    %51 = stablehlo.multiply %50, %49 : tensor<8x256xf32>
    %52 = stablehlo.add %47, %51 : tensor<8x256xf32>
    %cst_12 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
    %54 = stablehlo.multiply %53, %52 : tensor<8x256xf32>
    %55 = stablehlo.tanh %54 : tensor<8x256xf32>
    %cst_13 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
    %57 = stablehlo.add %56, %55 : tensor<8x256xf32>
    %cst_14 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_14, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
    %59 = stablehlo.multiply %58, %57 : tensor<8x256xf32>
    %60 = stablehlo.multiply %47, %59 : tensor<8x256xf32>
    %61 = stablehlo.dot_general %60, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x256xf32>, tensor<256x64xf32>) -> tensor<8x64xf32>
    %62 = stablehlo.add %46, %61 : tensor<8x64xf32>
    return %62 : tensor<8x64xf32>
  }
  func.func private @tril(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x8xi32>
    %2 = stablehlo.add %0, %1 : tensor<8x8xi32>
    %3 = stablehlo.iota dim = 1 : tensor<8x8xi32>
    %4 = stablehlo.compare GE, %2, %3, SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
    %6 = stablehlo.select %4, %arg0, %5 : tensor<8x8xi1>, tensor<8x8xf32>
    return %6 : tensor<8x8xf32>
  }
  func.func private @_where(%arg0: tensor<8x8xi1>, %arg1: tensor<8x8xf32>, %arg2: tensor<f32>) -> tensor<8x8xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<8x8xi1>, tensor<8x8xf32>
    return %2 : tensor<8x8xf32>
  }
}
