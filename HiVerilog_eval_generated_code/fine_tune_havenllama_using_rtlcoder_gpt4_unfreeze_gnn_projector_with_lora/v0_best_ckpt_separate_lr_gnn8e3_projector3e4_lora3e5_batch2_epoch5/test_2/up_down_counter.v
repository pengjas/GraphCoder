module up_down_counter (
    input wire clk,
    input wire reset,
    input wire up_down,
    output wire [15:0] count
);
16-bit counter logic

module counter_logic (
    input wire clk,
    input wire reset,
    input wire up_down,
    output wire [15:0] next_count
);

reg [15:0] count_reg;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        count_reg <= 16'h0000;
    end else begin
        if (up_down) begin
            count_reg <= count_reg + 16'h0001;
        end else begin
            count_reg <= count_reg - 16'h0001;
        end
    end
end

assign next_count = count_reg;

endmodule


module counter_register (
    input wire clk,
    input wire reset,
    input wire [15:0] next_count,
    output reg [15:0] count
);

always @(posedge clk or posedge reset) begin
    if (reset) begin
        count <= 16'h0000;
    end else begin
        count <= next_count;
    end
end

endmodule

endmodule