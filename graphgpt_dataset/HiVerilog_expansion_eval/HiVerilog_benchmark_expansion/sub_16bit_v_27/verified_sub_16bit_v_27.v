module add_16bit(A, B, S, C_out);
    input [16:1] A;       // 16-bit input A
    input [16:1] B;       // 16-bit input B
    output [16:1] S;      // 16-bit sum output
    output C_out;         // Carry out

    wire c4, c8, c12;

    // Instantiate 4-bit adders
    adder_4 add1(
        .A(A[4:1]),
        .B(B[4:1]),
        .C_in(0),
        .S(S[4:1]),
        .C_out(c4)
    );

    adder_4 add2(
        .A(A[8:5]),
        .B(B[8:5]),
        .C_in(c4),
        .S(S[8:5]),
        .C_out(c8)
    );

    adder_4 add3(
        .A(A[12:9]),
        .B(B[12:9]),
        .C_in(c8),
        .S(S[12:9]),
        .C_out(c12)
    );

    adder_4 add4(
        .A(A[16:13]),
        .B(B[16:13]),
        .C_in(c12),
        .S(S[16:13]),
        .C_out(C_out)
    );
endmodule

module adder_4(A, B, C_in, S, C_out);
    input [4:1] A;        // 4-bit input A
    input [4:1] B;        // 4-bit input B
    input C_in;           // Carry in
    output [4:1] S;       // 4-bit sum output
    output C_out;         // Carry out

    wire g1, g2, g3, g4;  // Generate signals
    wire p1, p2, p3, p4;  // Propagate signals
    wire c1, c2, c3;      // Internal carry signals

    // Instantiate full adders for each bit
    full_adder fa1(
        .A(A[1]),
        .B(B[1]),
        .C_in(C_in),
        .S(S[1]),
        .C_out(c1)
    );

    full_adder fa2(
        .A(A[2]),
        .B(B[2]),
        .C_in(c1),
        .S(S[2]),
        .C_out(c2)
    );

    full_adder fa3(
        .A(A[3]),
        .B(B[3]),
        .C_in(c2),
        .S(S[3]),
        .C_out(c3)
    );

    full_adder fa4(
        .A(A[4]),
        .B(B[4]),
        .C_in(c3),
        .S(S[4]),
        .C_out(C_out)
    );

    // Generate and propagate signals
    assign p1 = A[1] ^ B[1];
    assign g1 = A[1] & B[1];

    assign p2 = A[2] ^ B[2];
    assign g2 = A[2] & B[2];

    assign p3 = A[3] ^ B[3];
    assign g3 = A[3] & B[3];

    assign p4 = A[4] ^ B[4];
    assign g4 = A[4] & B[4];

endmodule

module full_adder(A, B, C_in, S, C_out);
    input A;              // Input A
    input B;              // Input B
    input C_in;           // Carry in
    output S;             // Sum output
    output C_out;         // Carry out

    assign S = A ^ B ^ C_in; // Sum calculation
    assign C_out = (A & B) | ((A ^ B) & C_in); // Carry out calculation
endmodule
