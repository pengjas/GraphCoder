module sub_8bit(A, B, D, B_out);
wire [3:0] carry_out;
    wire [7:0] B_inv;
    assign B_inv = ~B;

    full_adder fa0 (.A(A[0]), .B(B_inv[0]), .Cin(1'b0), .Dout(D[0]), .Cout(carry_out[0]));
    full_adder fa1 (.A(A[1]), .B(B_inv[1]), .Cin(carry_out[0]), .Dout(D[1]), .Cout(carry_out[1]));
    full_adder fa2 (.A(A[2]), .B(B_inv[2]), .Cin(carry_out[1]), .Dout(D[2]), .Cout(carry_out[2]));
    full_adder fa3 (.A(A[3]), .B(B_inv[3]), .Cin(carry_out[2]), .Dout(D[3]), .Cout(carry_out[3]));
    full_adder fa4 (.A(A[4]), .B(B_inv[4]), .Cin(carry_out[3]), .Dout(D[4]), .Cout(carry_out[4]));
    full_adder fa5 (.A(A[5]), .B(B_inv[5]), .Cin(carry_out[4]), .Dout(D[5]), .Cout(carry_out[5]));
    full_adder fa6 (.A(A[6]), .B(B_inv[6]), .Cin(carry_out[5]), .Dout(D[6]), .Cout(carry_out[6]));
    full_adder fa7 (.A(A[7]), .B(B_inv[7]), .Cin(carry_out[6]), .Dout(D[7]), .Cout(carry_out[7]));

    assign B_out = carry_out[7];

endmodule