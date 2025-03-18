module mult_8bit(X, Y, P);
    input [7:0] X;    // 8-bit input operand X
    input [7:0] Y;    // 8-bit input operand Y
    output [15:0] P;  // 16-bit output representing the product X * Y

    wire [15:0] products[0:7];  // Intermediate products for each bit of Y

    // Instantiate gen_product modules for each bit of Y
    gen_product GP0(.X(X), .Y_bit(Y[0]), .Shift(0), .Product(products[0]));
    gen_product GP1(.X(X), .Y_bit(Y[1]), .Shift(1), .Product(products[1]));
    gen_product GP2(.X(X), .Y_bit(Y[2]), .Shift(2), .Product(products[2]));
    gen_product GP3(.X(X), .Y_bit(Y[3]), .Shift(3), .Product(products[3]));
    gen_product GP4(.X(X), .Y_bit(Y[4]), .Shift(4), .Product(products[4]));
    gen_product GP5(.X(X), .Y_bit(Y[5]), .Shift(5), .Product(products[5]));
    gen_product GP6(.X(X), .Y_bit(Y[6]), .Shift(6), .Product(products[6]));
    gen_product GP7(.X(X), .Y_bit(Y[7]), .Shift(7), .Product(products[7]));

    // Sum all partial products to form the final product
    assign P = products[0] + products[1] + products[2] + products[3] +
               products[4] + products[5] + products[6] + products[7];

endmodule

module gen_product(X, Y_bit, Shift, Product);
    input [7:0] X;     // 8-bit input operand X
    input Y_bit;       // Single bit of input operand Y
    input [2:0] Shift; // Shift amount for partial product alignment
    output [15:0] Product; // Output partial product

    // Calculate partial product
    wire [7:0] partial_product = X & {8{Y_bit}};

    // Shift partial product to align according to bit position
    assign Product = partial_product << Shift;

endmodule
