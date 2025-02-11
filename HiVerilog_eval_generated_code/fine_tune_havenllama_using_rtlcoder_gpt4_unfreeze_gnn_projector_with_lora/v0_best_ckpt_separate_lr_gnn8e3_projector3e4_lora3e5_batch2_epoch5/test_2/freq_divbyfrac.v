module freq_divbyfrac(
    input               clk,
    input               rst_n,
    output              clk_div
);

parameter MUL2_DIV_CLK = 7;


reg [2:0] counter;
reg [2:0] clk_average;
reg [2:0] clk_adjustment;


always @(clk or rst_n) begin
    if (rst_n == 1'b0) begin
        counter <= 3'b0;
        clk_average <= 3'b0;
        clk_adjustment <= 3'b0;
    end else begin
        if (counter == MUL2_DIV_CLK - 1) begin
            counter <= 3'b0;
        end else begin
            counter <= counter + 3'b1;
        end

        clk_average <= {counter[2], counter[2], counter[2], counter[2]};
        clk_adjustment <= {counter[2], counter[2], counter[2]};
    end
end

assign clk_div = clk_average[0] | clk_adjustment[0];

endmodule