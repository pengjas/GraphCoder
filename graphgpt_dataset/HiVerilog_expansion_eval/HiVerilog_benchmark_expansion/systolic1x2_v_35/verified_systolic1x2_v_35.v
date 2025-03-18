module dual_shift_register(
    input clk,
    input rst,
    input load0,
    input load1,
    input [7:0] data0,
    input [7:0] data1,
    input shift0,
    input shift1,
    output [7:0] out0,
    output [7:0] out1
);
    ShiftRegister sr0 (
        .clk(clk),
        .rst(rst),
        .load(load0),
        .data(data0),
        .shift(shift0),
        .out(out0)
    );
    ShiftRegister sr1 (
        .clk(clk),
        .rst(rst),
        .load(load1),
        .data(data1),
        .shift(shift1),
        .out(out1)
    );

endmodule

module ShiftRegister (
    input clk,
    input rst,
    input load,
    input [7:0] data,
    input shift,
    output reg [7:0] out
);
    always @(posedge clk or posedge rst) begin
        if (rst)
            out <= 0;
        else if (load)
            out <= data;
        else if (shift)
            out <= out >> 1; // shift right operation
    end
endmodule