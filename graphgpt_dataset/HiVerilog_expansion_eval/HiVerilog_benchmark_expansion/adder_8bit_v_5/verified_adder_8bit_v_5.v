module multiplier_4bit(
    input [3:0] a, b,
    output [7:0] product);

    wire [3:0] partial_products[3:0];
    wire [7:0] sum[3:0];
    
    generate_partial_products GPP0 (.a(a), .b(b[0]), .partial_product(partial_products[0]));
    generate_partial_products GPP1 (.a(a), .b(b[1]), .partial_product(partial_products[1]));
    generate_partial_products GPP2 (.a(a), .b(b[2]), .partial_product(partial_products[2]));
    generate_partial_products GPP3 (.a(a), .b(b[3]), .partial_product(partial_products[3]));

    assign sum[0] = {4'b0, partial_products[0]};
    
    adder_4bit_with_shift A4BS0 (.a(sum[0]), .b({3'b0, partial_products[1], 1'b0}), .sum(sum[1]));
    adder_4bit_with_shift A4BS1 (.a(sum[1]), .b({2'b0, partial_products[2], 2'b0}), .sum(sum[2]));
    adder_4bit_with_shift A4BS2 (.a(sum[2]), .b({1'b0, partial_products[3], 3'b0}), .sum(sum[3]));

    assign product = sum[3];
endmodule

module generate_partial_products(input [3:0] a, input b, output [3:0] partial_product);
    assign partial_product = b ? a : 4'b0;
endmodule

module adder_4bit_with_shift(input [7:0] a, input [7:0] b, output [7:0] sum);
    wire [8:0] carry;
    assign {carry[0], sum[0]} = a[0] + b[0];
    assign {carry[1], sum[1]} = a[1] + b[1] + carry[0];
    assign {carry[2], sum[2]} = a[2] + b[2] + carry[1];
    assign {carry[3], sum[3]} = a[3] + b[3] + carry[2];
    assign {carry[4], sum[4]} = a[4] + b[4] + carry[3];
    assign {carry[5], sum[5]} = a[5] + b[5] + carry[4];
    assign {carry[6], sum[6]} = a[6] + b[6] + carry[5];
    assign {carry[7], sum[7]} = a[7] + b[7] + carry[6];
endmodule