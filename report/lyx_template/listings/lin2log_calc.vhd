	-- This is only a snippet for the demonstration of code listings in LyX and LaTeX
	
	
    log_scaled <= ld_snr_log10*to_unsigned(48,LOG_CORR_WIDTH);
    
    pipe_reg_log10: process(clk, rst) is --Aufteilung der log10-Stufe
	begin
		if rst = '1' then
            log_scaled_piped <= (others => '0');
            q_error_log10_piped <= (others => '0');
            carr_index_log10_piped <= (others => '0');
            valid_log10_piped <= '0';
            corrector_piped <= (others => '0');
		elsif clk'event and clk = '1' then
            corrector_piped <= corrector;
			log_scaled_piped <= log_scaled;
			q_error_log10_piped <= q_error_log10;
			carr_index_log10_piped <= carr_index_log10; 
			valid_log10_piped <= valid_log10; 
		end if;
	end process pipe_reg_log10;
    
    log_appx <= 1105-log_scaled_piped;
    
make_positive:process(log_appx,corrector_piped, q_error_log10_piped) is
begin
	if log_appx = 49 AND unsigned(q_error_log10_piped) =15 then -- read: unsigned(corrector) > to_unsigned(49,corrector'length)
		snr_fixpoint <= (others => '0'); -- lower resolution limit, otherwise overflow for negative SNR
	else
		snr_fixpoint <= log_appx-unsigned(corrector_piped); -- fixed-point result, 4 binary fractional digits
	end if;
end process make_positive;

fraction_log10 <= snr_fixpoint(3 downto 0); -- fractional part
log_snr <= std_logic_vector(shift_right(snr_fixpoint(SNR_WORD_WIDTH-1 downto 0),4)); -- integert part

	
	out_reg:process(clk,rst) is	
	begin
	  if rst = '1' then
	    log_snr_out <= (others => '0');
			fraction_out <= (others => '0');
			carr_index_out <= (others => '0');
			valid_out <= '0';
			snr_frac <= (others => '0');
 		elsif clk'event and clk = '1' then
			log_snr_out <= log_snr;
			fraction_out <= fraction_log10;
			snr_frac <= std_logic_vector(snr_fixpoint(SNR_WORD_WIDTH-1 downto 0));
			carr_index_out <= carr_index_log10_piped; 
			valid_out <= valid_log10_piped; 
		end if;
	end process out_reg;
	
end architecture Behavioral;

