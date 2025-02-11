module sub_64bit(A, B, D, B_out);
    input [63:0] A;
    input [3:0] B;
    output [63:0] D;
    output [3:0] B_out;

    wire [15:0] A_sub0, A_sub1, A_sub2, A_sub3;
    wire [7:0] B_sub0, B_sub1, B_sub2, B_sub3;
    wire [15:0] D_sub0, D_sub1, D_sub2, D_sub3;
    wire [7:0] B_out_sub0, B_out_sub1, B_out_sub2, B_out_sub3;

    submodule sub0(.A(A[63:32]), .B(B[3:0]), .D(D_sub0), .B_out(B_out_sub0));
    submodule sub1(.A(A[31:0]), .B(B[3:0]), .D(D_sub1), .B_out(B_out_sub1));
    submodule sub2(.A(A[15:0]), .B(B[7:0]), .D(D_sub2), .B_out(B_out_sub2));
    submodule sub3(.A(A_sub2), .B(B_sub2), .D(D_sub3), .B_out(B_out_sub3));

    assign A_sub0 = A[31:0];
    assign A_sub1 = A[15:0];
    assign A_sub2 = A[23:0];
    assign A_sub3 = A[31:0];

    assign B_sub0 = B[3:0];
    assign B_sub1 = B[7:0];
    assign B_sub2 = B[15:0];
    assign B_sub3 = B[23:0];

    assign D_sub0 = D_sub2 - D_sub3;
    assign D_sub1 = D_sub0 - D_sub1;
    assign D_sub2 = D_sub1 - D_sub3;
    assign D_sub3 = D_sub2 - D_sub3;

    assign B_out_sub0 = B_out_sub2 - B_out_sub3;
    assign B_out_sub1 = B_out_sub0 - B_out_sub3;
    assign B_out_sub2 = B_out_sub1 - B_out_sub3;
    assign B_out_sub3 = B_out_sub2 - B_out_sub3;

endmodule

module submodule(A, B, D, B_out);
    input [15:0] A;
    input [7:0] B;
    output [15:0] D;
    output [7:0] B_out;

    wire [7:0] B_sub0, B_sub1;
    wire [15:0] D_sub0, D_sub1;

    submodule sub0(.A(A[15:8]), .B(B[7:0]), .D(D_sub0), .B_out(B_sub0));
    submodule sub1(.A(A[7:0]), .B(B[7:0]), .D(D_sub1), .B_out(B_sub1));

    assign D = D_sub1 - D_sub0;
    assign B_out = B_sub1 - B_sub0;
endmodule

module submodule(A, B, D, B_out);
    input [7:0] A;
    input [7:0] B;
    output [7:0] D;
    output [7:0] B_out;

    wire [7:0] B_sub0, B_sub1;
    wire [7:0] D_sub0, D_sub1;

    submodule sub0(.A(A[7:4]), .B(B[7:0]), .D(D_sub0), .B_out(B_sub0));
    submodule sub1(.A(
endmodule