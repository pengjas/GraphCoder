module freq_divbyeven(
    input clk,
    input rst_n,
    output clk_div
);
parameter NUM_DIV = 2;

    reg [3:0] counter;
    reg clk_div;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter <= 0;
            clk_div <= 0;
        end else begin
            if (counter == NUM_DIV - 1) begin
                counter <= 0;
            end else begin
                counter <= counter + 1;
            end
            clk_div <= ~clk_div;
        end
    end

endmodule