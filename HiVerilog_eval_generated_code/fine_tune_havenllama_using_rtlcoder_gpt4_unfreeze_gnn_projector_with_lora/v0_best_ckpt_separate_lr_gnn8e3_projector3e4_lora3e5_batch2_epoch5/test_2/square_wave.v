module square_wave(
    input clk,
    input [7:0] freq,
    output wave_out
);
reg wave_out;
    always @(posedge clk) begin
        wave_out <= ~wave_out;
    end
endmodule

module Counter(
    input clk,
    input enable,
    input [7:0] freq,
    output [7:0] count
); ASSISTANT:
    reg [7:0] count_reg;
    always @(posedge clk) begin
        if (enable) begin
            if (count_reg == (freq - 1)) begin
                count_reg <= 0;
            end
            else begin
                count_reg <= count_reg + 1;
            end
        end
    end
endmodule

module wave_generator(
    input clk,
    input [7:0] count,
    input [7:0] freq,
    output wave_out
); ASSISTANT:
    reg wave_out;
    always @(posedge clk) begin
        if (count == (freq - 1)) begin
            wave_out <= 1;
        end
        else if (count == 0) begin
            wave_out <= 0;
        end
    end
endmodule