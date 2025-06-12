module multiplier_4bit(
    input [3:0] a, b,
    output [7:0] product);

    wire [7:0] intermediate[3:0];

    shift_and_add SAA0(.a(a), .b(b[0]), .product(intermediate[0]));
    shift_and_add SAA1(.a(a), .b(b[1]), .product(intermediate[1]));
    shift_and_add SAA2(.a(a), .b(b[2]), .product(intermediate[2]));
    shift_and_add SAA3(.a(a), .b(b[3]), .product(intermediate[3]));

    assign product = intermediate[0] + (intermediate[1] << 1) + (intermediate[2] << 2) + (intermediate[3] << 3);

endmodule

module shift_and_add(input [3:0] a, input b, output [7:0] product);
    assign product = b ? {4'b0000, a} : 8'b00000000;
endmodule