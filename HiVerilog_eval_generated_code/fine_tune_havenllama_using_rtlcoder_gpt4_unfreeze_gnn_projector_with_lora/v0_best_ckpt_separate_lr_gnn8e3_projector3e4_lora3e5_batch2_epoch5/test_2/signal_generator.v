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
            wave_reg <= 5'b00000;
        end else begin
            case (state)
                2'b00: begin
                    wave_reg <= wave_reg + 5'b000001;
                    state <= 2'b01;
                end
                2'b01: begin
                    wave_reg <= wave_reg - 5'b000001;
                    state <= 2'b10;
                end
                2'b10: begin
                    wave_reg <= wave_reg + 5'b0000001;
                    state <= 2'b11;
                end
                2'b11: begin
                    wave_reg <= wave_reg - 5'b00001;
                    state <= 2'b00;
                end
            endcase
        end
    end

    assign wave = wave_reg;

endmodule