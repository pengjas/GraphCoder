module subtractor_32bit(A, B, Bin, D, Bout);
    input [32:1] A;
    input [32:1] B;
    input Bin;
    output [32:1] D;
    output Bout;

    wire [3:0] b;

    subtractor_8 SUB1(
        .A(A[8:1]),
        .B(B[8:1]),
        .Bin(Bin),
        .D(D[8:1]),
        .Bout(b[0])
    );

    subtractor_8 SUB2(
        .A(A[16:9]),
        .B(B[16:9]),
        .Bin(b[0]),
        .D(D[16:9]),
        .Bout(b[1])
    );

    subtractor_8 SUB3(
        .A(A[24:17]),
        .B(B[24:17]),
        .Bin(b[1]),
        .D(D[24:17]),
        .Bout(b[2])
    );

    subtractor_8 SUB4(
        .A(A[32:25]),
        .B(B[32:25]),
        .Bin(b[2]),
        .D(D[32:25]),
        .Bout(b[3])
    );

    assign Bout = b[3];
endmodule

module subtractor_8(A, B, Bin, D, Bout);
    input [8:1] A;
    input [8:1] B;
    input Bin;
    output [8:1] D;
    output Bout;

    wire [7:0] borrow;

    full_subtractor FS1(.a(A[1]), .b(B[1]), .bin(Bin), .d(D[1]), .bout(borrow[0]));
    full_subtractor FS2(.a(A[2]), .b(B[2]), .bin(borrow[0]), .d(D[2]), .bout(borrow[1]));
    full_subtractor FS3(.a(A[3]), .b(B[3]), .bin(borrow[1]), .d(D[3]), .bout(borrow[2]));
    full_subtractor FS4(.a(A[4]), .b(B[4]), .bin(borrow[2]), .d(D[4]), .bout(borrow[3]));
    full_subtractor FS5(.a(A[5]), .b(B[5]), .bin(borrow[3]), .d(D[5]), .bout(borrow[4]));
    full_subtractor FS6(.a(A[6]), .b(B[6]), .bin(borrow[4]), .d(D[6]), .bout(borrow[5]));
    full_subtractor FS7(.a(A[7]), .b(B[7]), .bin(borrow[5]), .d(D[7]), .bout(borrow[6]));
    full_subtractor FS8(.a(A[8]), .b(B[8]), .bin(borrow[6]), .d(D[8]), .bout(Bout));
endmodule

module full_subtractor(a, b, bin, d, bout);
    input a, b, bin;
    output d, bout;

    assign d = a ^ b ^ bin;
    assign bout = (~a & b) | (bin & (~a ^ b));
endmodule
