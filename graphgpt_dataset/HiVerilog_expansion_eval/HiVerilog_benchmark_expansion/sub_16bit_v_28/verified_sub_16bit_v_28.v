module parity_16bit(X, Y, P);
    input [7:0] X;        // 8-bit input X
    input [7:0] Y;        // 8-bit input Y
    output P;             // Output parity

    wire P_X, P_Y;

    // Instantiate 8-bit parity generators
    parity_8 gen1(
        .data(X),
        .parity(P_X)
    );

    parity_8 gen2(
        .data(Y),
        .parity(P_Y)
    );

    // Final parity calculation to combine parities from X and Y
    assign P = P_X ^ P_Y;

endmodule

module parity_8(data, parity);
    input [7:0] data;     // 8-bit data input
    output parity;        // Output parity

    assign parity = ^data; // XOR of all bits to calculate even parity

endmodule
