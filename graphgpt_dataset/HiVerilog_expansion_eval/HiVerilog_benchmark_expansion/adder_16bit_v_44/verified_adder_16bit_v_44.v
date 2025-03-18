module multiplier_32bit (
    input wire [15:0] x,
    input wire [15:0] y,
    output wire [31:0] prod
);

    wire [15:0] p1, p2, p3, p4;
    wire [23:0] temp1, temp2, temp3, temp4;

    mult8 mult8_inst1 (
        .a(x[7:0]),
        .b(y[7:0]),
        .p(p1)
    );

    mult8 mult8_inst2 (
        .a(x[7:0]),
        .b(y[15:8]),
        .p(p2)
    );

    mult8 mult8_inst3 (
        .a(x[15:8]),
        .b(y[7:0]),
        .p(p3)
    );

    mult8 mult8_inst4 (
        .a(x[15:8]),
        .b(y[15:8]),
        .p(p4)
    );

    assign temp1 = {8'b0, p1};
    assign temp2 = {p2, 8'b0};
    assign temp3 = {p3, 8'b0};
    assign temp4 = {p4, 16'b0};

    assign prod = temp1 + temp2 + temp3 + temp4;

endmodule

module mult8 (
    input wire [7:0] a,
    input wire [7:0] b,
    output wire [15:0] p
);

    wire [15:0] partial [7:0];

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : bit_multiply
            assign partial[i] = (b[i] ? (a << i) : 16'b0);
        end
    endgenerate

    assign p = partial[0] + partial[1] + partial[2] + partial[3] + partial[4] + partial[5] + partial[6] + partial[7];

endmodule
