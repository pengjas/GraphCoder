module sub_32bit(A, B, Diff, B_out);

input [31:0] A;
input [1:0] B;
output [31:0] Diff;
output [1:0] B_out;

wire [15:0] Diff_16;
wire [1:0] B_out_1;

sub_16bit S1(A[15:0], B[1:0], Diff_16, B_out_1);
sub_16bit S2(A[31:16], B[1:0], Diff_16, B_out_1);

assign Diff = {Diff_16, B_out_1};
assign B_out = B_out_1;

endmodule

module sub_16bit(A, B, Diff, B_out);

input [15:0] A;
input [1:0] B;
output [15:0] Diff;
output [1:0] B_out;

wire [7:0] Diff_8;
wire [1:0] B_out_1;

sub_8bit S1(A[7:0], B[1:0], Diff_8, B_out_1);
sub_8bit S2(A[15:8], B[1:0], Diff_8, B_out_1);

assign Diff = {Diff_8, B_out_1};
assign B_out = B_out_1;

endmodule

module sub_8bit(A, B, Diff, B_out);

input [7:0] A;
input [1:0] B;
output [7:0] Diff;
output [1:0] B_out;

wire [3:0] Diff_4;
wire [1:0] B_out_1;

sub_4bit S1(A[3:0], B[1:0], Diff_4, B_out_1);
sub_4bit S2(A[7:4], B[1:0], Diff_4, B_out_1);
sub_4bit S3(A[11:8], B[1:0], Diff_4, B_out_1);
sub_4bit S4(A[15:12], B[1:0], Diff_4, B_out_1);

assign Diff = {Diff_4, B_out_1};
assign B_out = B_out_1;

endmodule

module sub_4bit(A, B, Diff, B_out);

input [3:0] A;
input [1:0] B;
output [3:0] Diff;
output [1:0] B_out;

wire [1:0] B_out_1;

sub_2bit S1(A[1:0], B[1:0], Diff[1:0], B_out_1);
sub_2bit S2(A[3:2], B[1:0], Diff[3:2], B_out_1);

assign Diff = {Diff[3:2], B_out_1};
assign B_out = B_out_1;

endmodule

module sub_2bit(A, B, Diff, B_out);

input [1:0] A;
input [1:0] B;
output [1:0] Diff;
output B_out;

assign Diff = A - B;
assign B_out = A[1] ^ B[1];

endmodule