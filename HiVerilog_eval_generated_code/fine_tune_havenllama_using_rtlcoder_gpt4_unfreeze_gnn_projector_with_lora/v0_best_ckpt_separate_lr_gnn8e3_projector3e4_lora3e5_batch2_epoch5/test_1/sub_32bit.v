module sub_32bit(A, B, Diff, B_out);
input [31:0] A, B;
output [15:0] Diff, B_out;
wire [15:0] S1, S2;

submodule S1(
    input [31:0] A, B,
    output [15:0] Diff, B_out
);

assign Diff = A - B;
assign B_out = B;

endsubmodule

submodule S2(
    input [31:0] A, B,
    output [15:0] Diff, B_out
);

assign Diff = A - B;
assign B_out = B;

endsubmodule

S1 s1_inst(A, B, Diff, B_out);
S2 s2_inst(A, B, Diff, B_out);

endmodule