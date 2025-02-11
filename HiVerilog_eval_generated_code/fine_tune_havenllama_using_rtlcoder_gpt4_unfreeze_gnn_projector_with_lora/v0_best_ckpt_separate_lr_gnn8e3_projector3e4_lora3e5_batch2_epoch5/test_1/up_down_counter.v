module counter_logic (
    input wire clk,
    input wire reset,
    input wire up_down,
    output wire [15:0] count
);
    reg [15:0] count_reg;

    always @(posedge clk) begin
        if (reset) begin
            count_reg <= 16'b0;
        end else begin
            if (up_down) begin
                count_reg <= count_reg + 1;
            end else begin
                count_reg <= count_reg - 1;
            end
        end
    end

    assign count = count_reg;
endmodule

module counter_register (
    input wire clk,
    input wire reset,
    input wire [15:0] count,
    output reg [15:0] count_reg
);

    always @(posedge clk) begin
        if (reset) begin
            count_reg <= 6'b0;
        end else begin
            count_reg <= count;
        end
    end
endmodule

module up_down_counter (
    input wire clk,
    input wire reset,
    input wire up_down,
    output reg [15:0] count
);
    wire [15:0] next_count;

    counter_logic counter_logic (
        .clk(clk),
        .reset(reset),
        .up_down(up_down),
        .count(next_count)
    );

    counter_register counter_register (
        .clk(clk),
        .reset(reset),
        .count(next_count),
        .count_reg(count)
    );
endmodule