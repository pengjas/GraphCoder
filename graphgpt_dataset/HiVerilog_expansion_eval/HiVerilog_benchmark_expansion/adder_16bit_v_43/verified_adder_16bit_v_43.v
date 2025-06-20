module comparator_32bit (
    input wire [31:0] a,
    input wire [31:0] b,
    output wire equal
);

    wire [3:0] equal_parts;

    compare8 comp8_inst1 (
        .a(a[31:24]),
        .b(b[31:24]),
        .equal(equal_parts[3])
    );

    compare8 comp8_inst2 (
        .a(a[23:16]),
        .b(b[23:16]),
        .equal(equal_parts[2])
    );

    compare8 comp8_inst3 (
        .a(a[15:8]),
        .b(b[15:8]),
        .equal(equal_parts[1])
    );

    compare8 comp8_inst4 (
        .a(a[7:0]),
        .b(b[7:0]),
        .equal(equal_parts[0])
    );

    assign equal = &(equal_parts);

endmodule

module compare8 (
    input wire [7:0] a,
    input wire [7:0] b,
    output wire equal
);

    assign equal = (a == b);

endmodule
