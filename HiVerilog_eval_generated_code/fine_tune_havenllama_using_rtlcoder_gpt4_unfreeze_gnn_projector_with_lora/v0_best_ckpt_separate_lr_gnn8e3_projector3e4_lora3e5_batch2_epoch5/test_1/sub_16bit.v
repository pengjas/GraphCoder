module sub_16bit(A, B, D, B_out);
input [16:1] A, B;
output [16:1] D;
output B_out;

wire [4:1] B_out_1, B_out_2, B_out_3;
wire [4:1] D_1, D_2, D_3;

submodule sub4bit(A, B, D, B_out);
input [4:1] A, B;
output [4:1] D;
output B_out;

wire [2:1] B_out_1, B_out_2;
wire [2:1] D_1, D_2;

submodule sub2bit(A, B, D, B_out);
input [2:1] A, B;
output [2:1] D;
output B_out;

wire [1:1] B_out_1, B_out_2;
wire [1:1] D_1, D_1;

assign B_out_1 = (A[1] & ~ B[1]) | (A[1] & B[1]);
assign B_out_1 = (A[2] & ~B[2]) | (A[2] & B[2]);
assign B_out_2 = (A[3] & ~B[3]) | (A[3] & B[3]);
assign B_out_3 = (A[4] & ~B[4]) | (A[4] & B[4]);

assign D_1 = (A[2] & ~B[2]) | (A[2] & B[2]);
assign D_2 = (A[3] & ~B[3]) | (A[3] & B[3]);
assign D_3 = (A[4] & ~B[4]) | (A[4] & B[4]);

assign B_out = B_out_1 | B_out_2 | B_out_3;
assign D = D_1 | D_2 | D_3;

endmodule

submodule sub2bit(A, B, D, B_out);
input [2:1] A, B;
output [2:1] D;
output B_out;

wire [1:1] B_out_1, B_out_2;
wire [1:1] D_1, D_2;

assign B_out_1 = (A[1] & ~B[1]) | (A[1] & B[1]);
assign B_out_2 = (A[2] & ~B[2]) | (A[2] & B[2]);

assign D_1 = (A[1] & ~B[1]) | (A[1] & B[1]);
assign D_2 = (A[2] & ~B[2]) | (A[2] & B[2]);

assign B_out = B_out_1 | B_out_2;
assign D = D_1 | D_2;

endmodule

submodule sub4bit(A, B, D, B_out);
input [4:1] A, B;
output [4:1] D;
output B_out;

wire [2:1] B_out_1, B_out_2;
wire [2:1] D_1, D_2;

submodule sub2bit(A, B, D, B_out);
input [2:1] A, B;
output [2:1] D;
output B_out;

wire [1:1] B_out_1, B_out_2;
wire [1:1] D_1, D_2;

assign B_out_1 = (A[1] & ~B[1]) | (A[1] & B[1]);
assign B_out_2 = (A[2] & ~B[2]) | (A[2] & B[2]);

assign D_1 = (A[1] & ~B[1]) | (A[1] & B[1]);
assign D_2 = (A[2] & ~B[2]) | (A[2] & B[2]);

assign B_out = B_out_1 | B_out_
endmodule