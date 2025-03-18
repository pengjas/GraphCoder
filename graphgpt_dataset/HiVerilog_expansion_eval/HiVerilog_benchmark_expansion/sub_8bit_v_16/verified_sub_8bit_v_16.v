module mul_4bit(X, Y, P);
    input [3:0] X;    // 4-bit input operand X
    input [3:0] Y;    // 4-bit input operand Y
    output [7:0] P;   // 8-bit output product

    wire [7:0] pp0, pp1, pp2, pp3; // partial products
    wire [7:0] sum1, sum2, sum3;   // intermediate sums
    wire cout1, cout2, cout3;      // carry outputs from adders

    // Generate partial products
    assign pp0 = Y[0] ? {4'b0000, X} : 8'b00000000;
    assign pp1 = Y[1] ? {3'b000, X, 1'b0} : 8'b00000000;
    assign pp2 = Y[2] ? {2'b00, X, 2'b00} : 8'b00000000;
    assign pp3 = Y[3] ? {1'b0, X, 3'b000} : 8'b00000000;

    // Add partial products
    adder_8bit add1(.A(pp0), .B(pp1), .D(sum1), .Cout(cout1));
    adder_8bit add2(.A(sum1), .B(pp2), .D(sum2), .Cout(cout2));
    adder_8bit add3(.A(sum2), .B(pp3), .D(P), .Cout(cout3));

endmodule

module adder_8bit(A, B, D, Cout);
    input [7:0] A;     // 8-bit input operand A
    input [7:0] B;     // 8-bit input operand B
    output [7:0] D;    // 8-bit output representing the sum
    output Cout;       // Carry out

    wire [7:0] sum;    // Intermediate sum values
    wire [7:0] carry;  // Intermediate carry values

    // Instantiate full adders
    full_adder FA0(.A(A[0]), .B(B[0]), .Cin(1'b0), .Sum(sum[0]), .Cout(carry[0]));
    full_adder FA1(.A(A[1]), .B(B[1]), .Cin(carry[0]), .Sum(sum[1]), .Cout(carry[1]));
    full_adder FA2(.A(A[2]), .B(B[2]), .Cin(carry[1]), .Sum(sum[2]), .Cout(carry[2]));
    full_adder FA3(.A(A[3]), .B(B[3]), .Cin(carry[2]), .Sum(sum[3]), .Cout(carry[3]));
    full_adder FA4(.A(A[4]), .B(B[4]), .Cin(carry[3]), .Sum(sum[4]), .Cout(carry[4]));
    full_adder FA5(.A(A[5]), .B(B[5]), .Cin(carry[4]), .Sum(sum[5]), .Cout(carry[5]));
    full_adder FA6(.A(A[6]), .B(B[6]), .Cin(carry[5]), .Sum(sum[6]), .Cout(carry[6]));
    full_adder FA7(.A(A[7]), .B(B[7]), .Cin(carry[6]), .Sum(sum[7]), .Cout(Cout));

    // Assign the sum to output
    assign D = sum;

endmodule

module full_adder(A, B, Cin, Sum, Cout);
    input A;       // Input A
    input B;       // Input B
    input Cin;     // Carry input
    output Sum;    // Sum output
    output Cout;   // Carry output

    assign Sum = A ^ B ^ Cin;                  // Sum calculation
    assign Cout = (A & B) | (Cin & (A ^ B));  // Carry calculation
endmodule
