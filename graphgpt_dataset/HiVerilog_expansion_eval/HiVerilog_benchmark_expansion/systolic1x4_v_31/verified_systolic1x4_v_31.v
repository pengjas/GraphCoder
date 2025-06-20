module matrix_multiplier2x2(
    input clk,
    input rst,
    input [7:0] x0,
    input [7:0] x1,
    input [7:0] y0,
    input [7:0] y1,
    output [15:0] p0,
    output [15:0] p1,
    output [15:0] p2,
    output [15:0] p3
);
    // Instantiate multiplier units
    multiplier mult0(.clk(clk), .rst(rst), .a(x0), .b(y0), .result(p0));
    multiplier mult1(.clk(clk), .rst(rst), .a(x1), .b(y0), .result(p1));
    multiplier mult2(.clk(clk), .rst(rst), .a(x0), .b(y1), .result(p2));
    multiplier mult3(.clk(clk), .rst(rst), .a(x1), .b(y1), .result(p3));
endmodule

module multiplier(
    input clk,
    input rst,
    input [7:0] a,
    input [7:0] b,
    output reg [15:0] result
);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            result <= 0;
        end else begin
            result <= a * b;
        end
    end
endmodule
