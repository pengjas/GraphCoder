module dual_latch_system (
    input clk,
    input reset,
    input [7:0] d0,
    input [7:0] d1,
    input load0,
    input load1,
    output [7:0] q0,
    output [7:0] q1
);
    Latch latch0 (
        .clk(clk),
        .reset(reset),
        .d(d0),
        .load(load0),
        .q(q0)
    );
    Latch latch1 (
        .clk(clk),
        .reset(reset),
        .d(d1),
        .load(load1),
        .q(q1)
    );
endmodule

module Latch (
    input clk,
    input reset,
    input [7:0] d,
    input load,
    output reg [7:0] q
);
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            q <= 0;
        end else if (load) begin
            q <= d;
        end
    end
endmodule
