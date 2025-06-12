module mult_8bit(X, Y, P);
    input [7:0] X;      // 8-bit input operand X
    input [7:0] Y;      // 8-bit input operand Y
    output [15:0] P;    // 16-bit output representing the product X * Y

    wire [15:0] product[0:7]; // Partial products from each stage

    // Instantiate add_shift modules for each bit in Y
    add_shift AS0(.X(X), .shift(0), .Y_bit(Y[0]), .partial_sum(16'b0), .P(product[0]));
    add_shift AS1(.X(X), .shift(1), .Y_bit(Y[1]), .partial_sum(product[0]), .P(product[1]));
    add_shift AS2(.X(X), .shift(2), .Y_bit(Y[2]), .partial_sum(product[1]), .P(product[2]));
    add_shift AS3(.X(X), .shift(3), .Y_bit(Y[3]), .partial_sum(product[2]), .P(product[3]));
    add_shift AS4(.X(X), .shift(4), .Y_bit(Y[4]), .partial_sum(product[3]), .P(product[4]));
    add_shift AS5(.X(X), .shift(5), .Y_bit(Y[5]), .partial_sum(product[4]), .P(product[5]));
    add_shift AS6(.X(X), .shift(6), .Y_bit(Y[6]), .partial_sum(product[5]), .P(product[6]));
    add_shift AS7(.X(X), .shift(7), .Y_bit(Y[7]), .partial_sum(product[6]), .P(product[7]));

    // Assign the final product
    assign P = product[7];

endmodule

module add_shift(X, shift, Y_bit, partial_sum, P);
    input [7:0] X;         // Multiplicand
    input [2:0] shift;     // Shift amount
    input Y_bit;           // Specific bit of multiplier Y
    input [15:0] partial_sum; // Partial sum input
    output [15:0] P;       // Output partial product

    wire [15:0] shifted_X;

    // Shift X left by 'shift' positions
    assign shifted_X = X << shift;

    // Conditional addition based on Y_bit
    assign P = Y_bit ? partial_sum + shifted_X : partial_sum;

endmodule