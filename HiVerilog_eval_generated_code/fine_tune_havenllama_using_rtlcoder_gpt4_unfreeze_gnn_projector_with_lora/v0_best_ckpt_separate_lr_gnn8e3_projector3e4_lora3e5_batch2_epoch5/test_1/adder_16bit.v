module adder_16bit (
    input wire [15:0] a,
    input wire [15:0] b,
    input wire Cin,
    output wire [15:0] y,
    output wire Co
);

    
    adder_8bit adder_8bit_inst1 (
        .a(a[7:0]),
        .b(b[7:0]),
        .Cin(Cin),
        .y(y[7:0]),
        .Co(Co)
    );

    
    adder_8bit adder_8bit_inst2 (
        .a(a[15:8]),
        .b(b[15:8]),
        .Cin(Co),
        .y(y[15:8]),
        .Co(Co)
    );

endmodule


module adder_8bit (
    input wire [7:0] a,
    input wire [7:0] b,
    input wire Cin,
    output wire [7:0] y,
    output wire Co
);

    assign y = a + b + Cin;
    assign Co = (a[7] & b[7]) | (a[7] & Co) | (b[7] & Co);

endmodule