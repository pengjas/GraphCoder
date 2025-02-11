module adder_16bit (
    input wire [15:0] a,
    input wire [15:0] b,
    input wire Cin,
    output wire [15:0] y,
    output wire Co
);

    
    module adder_8bit (
        input wire [7:0] a,
        input wire [7:0] b,
        input wire Cin,
        output wire [7:0] y,
        output wire Co
    );
        wire [8:0] sum;
        assign sum = a + b + Cin;
        assign y = sum[7:0];
        assign Co = sum[8];
    endmodule

    
    module adder_16bit (
        input wire [15:0] a,
        input wire [15:0] b,
        input wire Cin,
        output wire [15:0] y,
        output wire Co
    );
        wire [16:0] sum;
        assign sum = a + b + Cin;
        assign y = sum[15:0];
        assign Co = sum[16];
    endmodule

    
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