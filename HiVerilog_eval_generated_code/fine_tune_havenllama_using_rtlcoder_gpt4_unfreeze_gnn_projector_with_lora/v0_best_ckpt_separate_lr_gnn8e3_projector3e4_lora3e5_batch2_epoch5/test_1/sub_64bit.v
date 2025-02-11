module sub_64bit(A, B, D, B_out);
    input [63:0] A;
    input [3:0] B;
    output [63:0] D;
    output [3:0] B_out;

    wire [15:0] A_out0;
    wire [15:0] A_out1;
    wire [15:0] A_out2;
    wire [15:0] A_out3;
    wire [7:0] B_out0;
    wire [7:0] B_out1;
    wire [7:0] B_out2;
    wire [7:0] B_out3;

    submodule sub0(A[63:48], B[3:0], D[15:0], B_out[3:0]);
        submodule sub1(A[47:32], B[3:0], D[15:0], B_out[3:0]);
            submodule sub2(A[39:16], B[3:0], D[15:0], B_out[3:0]);
                submodule sub3(A[31:0], B[3:0], D[7:0], B_out[3:0]);
                endmodule
            endmodule
        endmodule
    endmodule
    endmodule
endmodule

end
endmodule