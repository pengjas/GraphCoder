module freq_divbyodd(
    input clk,
    input rst_n,
    output clk_div
);
parameter NUM_DIV = 5;

    reg [NUM_DIV - 1:0] pos_edge_counter;
    reg [NUM_DIV - 1:0] neg_edge_counter;
    reg [NUM_DIV - 1:0] clk_div1;
    reg [NUM_DIV - 1:0] clk_div2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pos_edge_counter <= 0;
            neg_edge_counter <= 0;
            clk_div1 <= 0;
            clk_div2 <= 0;
        end else begin
            if (clk) begin
                pos_edge_counter <= pos_edge_counter + 1;
                neg_edge_counter <= neg_edge_counter + 1;
            end else begin
                pos_edge_counter <= pos_edge_counter;
                neg_edge_counter <= neg_edge_counter;
            end
        end
    end

    assign clk_div = clk_div1 | clk_div2;

endmodule