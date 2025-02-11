module freq_divbyfrac(
    input               clk,
    input               rst_n,
    output              clk_div
);

parameter MUL2_DIV_CLK = 7;


reg [2:0] counter;
reg [1:0] clk_average;
reg [1:0] clk_adjustment;


always @(clk or rst_n) begin
    if (~rst_n) begin
        counter <= counter + 1'b1;
    end else begin
        counter <= 2'b0;
    end
end

assign clk_average = (counter == 2'b00);

always @(clk or rst_n) begin
    if (~rst_n) begin
        clk_adjustment <= 2'b0;
    end else begin
        clk_adjustment <= {~clk_average, clk_average};
    end
end

assign clk_div = clk_adjustment[0] | clk_adjustment[1];

endmodule