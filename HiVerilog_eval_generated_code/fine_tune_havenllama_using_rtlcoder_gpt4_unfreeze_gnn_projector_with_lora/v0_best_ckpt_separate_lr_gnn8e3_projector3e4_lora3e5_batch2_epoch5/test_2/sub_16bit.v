module sub_16bit(A, B, D, B_out);
input [16:1] A, B;
output [16:1] D;
output B_out;
wire [3:1] B_out;

wire [3:1] B_out_0, B_out_1, B_out_2, B_out_3;

submodule sub4bit(A, B, D, B_out);
input [3:1] A, B;
output [3:1] D;
output B_out;

wire [3:1] B_out_0, B_out_1, B_out_2;

assign D[0] = A[0] ^ B[0];
assign D[1] = A[1] ^ B[1];
assign D[2] = A[2] ^ B[2];
assign D[3] = A[3] ^ B[3];

assign B_out_0 = A[0] & B[0];
assign B_out_1 = A[1] & B[1];
assign B_out_2 = A[2] & B[2];
assign B_out = A[3] & B[3];

endmodule

submodule sub1(A, B, D, B_out);
input [15:1] A, B;
output [15:1] D;
output B_out;

wire [3:1] B_out_0, B_out_1, B_out_2, B_out_3;

submodule sub4bit(A, B, D, B_out);
input [3:1] A, B;
output [3:1] D;
output B_out;

wire [3:1] B_out_0, B_out_1, B_out_2;

assign D[0] = A[0] ^ B[0];
assign D[1] = A[1] ^ B[1];
assign D[2] = A[2] ^ B[2];
assign D[3] = A[3] ^ B[3];

assign B_out_0 = A[0] & B[0];
assign B_out_1 = A[1] & B[1];
assign B_out_2 = A[2] & B[2];
assign B_out = A[3] & B[3];

endmodule

submodule sub2(A, B, D, B_out);
input [14:1] A, B;
output [14:1] D;
output B_out;

wire [3:1] B_out_0, B_out_1, B_out_2, B_out_3;

submodule sub4bit(A, B, D, B_out);
input [3:1] A, B;
output [3:1] D;
output B_out;

wire [3:1] B_out_0, B_out_1, B_out_2;

assign D[0] = A[0] ^ B[0];
assign D[1] = A[1] ^ B[1];
assign D[2] = A[2] ^ B[2];
assign D[3] = A[3] ^ B[3];

assign B_out_0 = A[0] & B[0];
assign B_out_1 = A[1] & B[1];
assign B_out_2 = A[2] & B[2];
assign B_out = A[3] & B[3];

endmodule

submodule sub3(A, B, D, B_out);
input [13:1] A, B;
output [13:1] D;
output B_out;

wire [3:1] B_out_0, B_out_1, B_out_2, B_out_3;

submodule sub4bit(A, B, D, B_out);
input [3:1] A, B;
output [3:1] D;
output B_out;

wire [3:1] B_out_0, B_out_1, B_out_2;

assign D[0] = A[0] ^ B[0];
endmodule