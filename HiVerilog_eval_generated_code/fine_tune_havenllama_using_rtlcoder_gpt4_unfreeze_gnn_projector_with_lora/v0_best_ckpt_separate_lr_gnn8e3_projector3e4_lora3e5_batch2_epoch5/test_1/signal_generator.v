module signal_generator(
  input clk,
  input rst_n,
  output reg [4:0] wave
);
reg [1:0] state;
    reg [4:0] wave_reg;

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            state <= 2'b00;
            wave_reg <= 5'b0;
        end else begin
            case (state)
                2'b00: begin
                    wave_reg <= wave_reg + 1;
                    if (wave_reg == 5'b31) begin
                        state <= 2'b01;
                    end
                end
                2'b01: begin
                    wave_reg <= wave_reg - 1;
                    if (wave_reg == 5'b0) begin
                        state <= 2'b00;
                    end
                end
            end
        endcase
    end

    assign wave = wave_reg;

endmodule